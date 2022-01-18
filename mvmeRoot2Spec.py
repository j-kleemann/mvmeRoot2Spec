#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright © 2021 Jörn Kleemann
#
# mvmeRoot2Spec is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# mvmeRoot2Spec is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with mvmeRoot2Spec. If not, see <https://www.gnu.org/licenses/>.
"""\
Sort events of a ROOT file exported by the mvme_root_client from a Mesytech VME DAQ into spectra (histograms).
Specifically aimed at the mvme DAQ used since 2021 at the High Intensity γ-ray Source (HIγS) facility, located at the Triangle Universities Nuclear Laboratory in Durham, NC, USA.
"""

# BACKLOG Distribute among multiple source files?
# BACKLOG Rename functions, variables etc
# BACKLOG improve typing of custom types

# CONSIDER: Implement 2D Hist Export?

# TODO Add options for building of timehists to mvmeRoot2SpecCLI

from __future__ import annotations
import abc
import itertools
import sys
import os
import math
import datetime
import argparse
from typing import Any, Callable, ClassVar, Final, Iterable, Iterator, Sequence, Literal, NewType, Optional, TextIO, Tuple, Union
import dataclasses
import contextlib
import re
import concurrent.futures
import numpy as np
import uproot
import hist

# Constants
RAW_STR: Final = "Raw"
CAL_STR: Final = "Cal"
DET_STR: Final = "Det"
ADDBACK_STR: Final = "Addback"
BINNING_STR: Final = "b"
ON_BEAM_GATED_APPENDIX_STR: Final = "OnBeamGated"
OFF_BEAM_GATED_APPENDIX_STR: Final = "OffBeamGated"
TO_RF_APPENDIX_STR: Final = "ToRF"
SORTED_STR: Final = "Sorted"
RUN_STR: Final = "Run"
FILE_EXT: Final = "txt"
CALLST_EXT: Final = "callst"

MvmeModuleElement = NewType("MvmeModuleElement", str) # e.g. "clovers/amplitude[16]"
MvmeDataBatch = dict[MvmeModuleElement, np.ndarray] # Shape of batch: (batchSize, channels)
ChannelNo = NewType("ChannelNo", int)
ChannelsDict = dict[MvmeModuleElement, Sequence[ChannelNo]]
Calibration = Union[Callable[[np.ndarray], np.ndarray], "TupleCalibration", tuple[float, ...]]
CalibDict = dict[str, Calibration]


class ExplicitCheckFailedError(Exception):
  "Exception raised to abort execution in this module whenever an explicit check/validation fails. Caught by mvmeRoot2SpecCLI to exit without printing a stack traceback."
  pass


class NoCalibrationFoundError(ExplicitCheckFailedError):
  "Exception raised when no calibration is found for construction of a calibrated histogram."
  pass


def iterateAhead(iterator: Iterable) -> Iterator[Any]:
  "BACKLOG."

  class StopIterationSentinel:
    "BACKLOG."

  iterator = iter(iterator) # If already an Iterator this should have no effect (every Iterator should also be an Iterable)
  nextVal = next(iterator)
  with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
    while nextVal is not StopIterationSentinel:
      future = executor.submit(next, iterator, StopIterationSentinel)
      yield nextVal
      nextVal = future.result()


class updateablePrint:
  """Basically a print function extended with the updatable keyword to overwrite/update successive prints on a terminal.
  Successive calls with updatable=True will overwrite each others last output line using ANSI terminal escapes, if a terminal was detected.
  """
  lastPrintUpdateable: bool = False # Used as a "static" variable of the function
  # Class could be improved in the follwing aspects:
  # Handle calls with different file keywords via a lastPrintUpdateable dict with files as keys
  # Optionally also allow the creation of fixed-file instances (still need the dict though as two instances could share the same file or need to return an existing instance in the constructor if one with that file was already created)
  # In any case it might also be hard to figure out whether to paths actually refer to the same file
  # Let updatable prints allow for non \n endlines and only insert the final ANSI escape as soon as a \n is read
  # Fix the typing mess with callable vs class and the name mess (not really a printF anymore, rather a printClass, but it never was a pure printF if it required a updatable keyword to start with...)
  # Maybe make an extra function cls.U(...) for updatable prints instead of the updatable keyword?

  @classmethod
  def __call__(cls, *args, updatable: bool = False, end="\n", **kw_args) -> None:
    "Call the updateable print function"
    if updatable and (("file" not in kw_args and sys.stdout.isatty()) or ("file" in kw_args and hasattr(kw_args["file"], "isatty") and callable(kw_args["file"].isatty) and kw_args["file"].isatty())):
      # If file/stdout is a console (i.e. not redirection to file/pipe), assume it can handle ANSI escapes and use them to
      # overwrite the previous line for progress/status outputs. Other possibility would be to use \r instead of \n
      # as line end, as \r still causes a stdout flush, but positions the cursor at the beginning of the same line,
      # causing the next print to overwrite each character one at a time (i.e. not clearing the whole line though)
      # Here the order of the ANSI escapes is carefully chosen, so that the cursor should never move above a non-updateable
      # line, so that even if regular not aware print()s get mixed in by accident (e.g. exceptions, warnings), these are
      # not overwritten, but instead overwrite the updateable line.
      print("\x1b[2K", end="", **kw_args) # ANSI escape to clear the current line
      print(*args, end="\n", **kw_args)
      print("\x1b[1F", end="", **kw_args) # ANSI escape to move the cursor to the beginning of the previous line, will usually not be flushed immediately though
      cls.lastPrintUpdateable = True
    else:
      cls.makeLastLineUnUpdatable(**kw_args)
      print(*args, end=end, **kw_args)

  @classmethod
  def makeLastLineUnUpdatable(cls, **kw_args):
    "BACKLOG."
    # Alternative: Just call cls(end="")
    if (cls.lastPrintUpdateable):
      print(end="\n", **kw_args) # If the last print was updateable the cursor will still be resting at the beginning of its line, hence move it down
      cls.lastPrintUpdateable = False


@dataclasses.dataclass(frozen=True)
class BinningSpec():
  "Container for uniform histogram binning specifications. Ensures compability of upperEdge to lowerEdge and binWidth and calculates corresponding nBins."
  lowerEdge: float
  upperEdge: float
  binWidth: float
  nBins: int = dataclasses.field(init=False)
  exceptionOnRoundUpperEdge: bool = dataclasses.field(default=True, repr=False, compare=False)

  @classmethod
  def constructCenterAlignedBinning(cls, upperEdge: float, binWidth: float, lowerEdge: float = 0, centerAlignmentValue: float = 0, exceptionOnRounding: bool = True) -> BinningSpec:
    "Constructs a BinningSpec with centerAlignmentValue being the center of a (potential) bin."
    return cls.constructEdgeAlignedBinning(upperEdge, binWidth, lowerEdge, centerAlignmentValue - binWidth / 2, exceptionOnRounding)

  @classmethod
  def constructEdgeAlignedBinning(cls, upperEdge: float, binWidth: float, lowerEdge: float = 0, edgeAlignmentValue: float = 0, exceptionOnRounding: bool = True) -> BinningSpec:
    "Constructs a BinningSpec with edgeAlignmentValue being an edge of a (potential) bin."
    alignedLowerEdge = math.floor((lowerEdge - edgeAlignmentValue) / binWidth) * binWidth + edgeAlignmentValue
    alignedUpperEdge = math.ceil((upperEdge - edgeAlignmentValue) / binWidth) * binWidth + edgeAlignmentValue
    if exceptionOnRounding and (alignedUpperEdge != upperEdge or alignedLowerEdge != lowerEdge):
      raise ValueError(f"Invalid BinningSpec: {upperEdge=} and/or {lowerEdge=} do not match {binWidth=} with {edgeAlignmentValue=}! Appropriate values would be upperEdge={alignedUpperEdge} and lowerEdge={alignedLowerEdge}.")
    return cls(alignedLowerEdge, alignedUpperEdge, binWidth)

  def __post_init__(self) -> None:
    "Called by __init__() generated by dataclasses.dataclass. Here used to check compability of upperEdge to lowerEdge and binWidth and calculating corresponding nBins."
    upperEdge = math.ceil((self.upperEdge - self.lowerEdge) / self.binWidth) * self.binWidth + self.lowerEdge
    if self.exceptionOnRoundUpperEdge and self.upperEdge != upperEdge:
      raise ValueError(f"Invalid BinningSpec: {self.upperEdge=} does not match {self.lowerEdge=} and {self.binWidth=}! An appropriate value for upperEdge would be {upperEdge}.")
    object.__setattr__(self, 'upperEdge', upperEdge) # Necessary due to dataclass(frozen=True)
    nBins = round((self.upperEdge - self.lowerEdge) / self.binWidth)
    object.__setattr__(self, 'nBins', nBins) # Necessary due to dataclass(frozen=True)

  def asHistAxisKW(self) -> dict[str, float]:
    "Returns a keyword dictionary suitable to be unpacked (using **) to the hist.axis.Regular constructor."
    return dict(bins=self.nBins, start=self.lowerEdge, stop=self.upperEdge)

  def getHDTVCalibrationStr(self) -> str:
    "BACKLOG"
    # "   ".join(map(str, self.getHDTVCalibration()))
    return f"{self.binWidth/2+self.lowerEdge}   {self.binWidth}"


class TupleCalibration(tuple):
  "BACKLOG."

  def __repr__(self) -> str:
    return f"{type(self).__name__}({super().__repr__()})"

  def __call__(self, data: np.ndarray) -> np.ndarray:
    "BACKLOG."
    result = np.full(data.shape, self[-1]) # Use Horner's method
    for coefficient in self[-2::-1]: # All elements but the last of self in reversed order
      result = result * data + coefficient
    return result


class ScaleCalibration(TupleCalibration):
  "BACKLOG."
  timeBinWidth: Final = 25 / 1024 # 25ns / 1024 was the used MDPP-16 TDC resolution, see also tdc_resolution in https://www.mesytec.com/products/datasheets/MDPP-16_SCP-RCP.pdf

  def __new__(cls, scalingFactor: float) -> ScaleCalibration:
    "BACKLOG."
    return super().__new__(cls, (0, scalingFactor))

  def __repr__(self) -> str:
    return f"{type(self).__name__}({self[1]})"

  def __call__(self, data: np.ndarray) -> np.ndarray:
    "BACKLOG."
    return data * self[1]

  @property
  def scalingFactor(self) -> float:
    "BACKLOG."
    return self[1]


@dataclasses.dataclass
class LazyCachedAugmentedModuleBatch():
  "BACKLOG."
  data: dict[tuple[MvmeModuleElement, Union[ChannelNo, tuple[ChannelNo, ...]], Union[Literal["NonNaNMask"], tuple[float, ...], tuple[tuple[float, ...], ...]]], np.ndarray]
  calibDict: dict[tuple[MvmeModuleElement, Union[ChannelNo, tuple[ChannelNo, ...]], Union[Literal["NonNaNMask"], tuple[float, ...], tuple[tuple[float, ...], ...]]], np.ndarray] = dataclasses.field(default_factory=dict)

  def __getitem__(self, key: Union[MvmeModuleElement, tuple[MvmeModuleElement, Union[ChannelNo, tuple[ChannelNo, ...]], Union[Literal[True, False, "NonNaNMask"], CalibDict, Calibration, tuple[Calibration, ...]]]]) -> np.ndarray:
    "BACKLOG"
    if isinstance(key, tuple):
      return self.getLazy(*key)
    return self.getLazy(key)

  def getLazy(self, mod: MvmeModuleElement, channelNoOrNos: Union[ChannelNo, tuple[ChannelNo, ...]] = 0, cal: Union[Literal[True, False, "NonNaNMask"], CalibDict, Calibration, tuple[Calibration, ...]] = False) -> np.ndarray:
    "BACKLOG."
    # Don't have to filter for NaNs as boost histogram just puts them in the overflow-bin by convention: https://www.boost.org/doc/libs/1_77_0/libs/histogram/doc/html/histogram/rationale.html#histogram.rationale.uoflow
    if cal is False:
      return self.data[mod][:, channelNoOrNos]
    if cal is True:
      cal = self.calibDict
    if not isinstance(channelNoOrNos, tuple): # Single channel data is requested
      if isinstance(cal, dict):
        cal = DataAccumulatorBase.findCalib(mod, channelNoOrNos, cal)
      if not (isinstance(cal, str) or callable(cal)):
        cal = TupleCalibration(cal)
      cacheable = (cal == "NonNaNMask" or isinstance(cal, TupleCalibration))
      if cacheable:
        with contextlib.suppress(KeyError):
          return self.data[(mod, channelNoOrNos, cal)]
      if cal == "NonNaNMask":
        returnV = ~np.isnan(self.data[mod][:, channelNoOrNos])
      else: # Calibrated data is requested
        data: np.ndarray = self.data[mod][:, channelNoOrNos]
        returnV = np.full(data.shape, np.nan)
        nonNaNMask = self.getLazy(mod, channelNoOrNos, "NonNaNMask")
        returnV[nonNaNMask] = cal(data[nonNaNMask])
    else: # Addback data/NonNaNMask are requested
      if isinstance(cal, dict):
        cal = tuple(DataAccumulatorBase.findCalib(mod, chNo, cal) for chNo in channelNoOrNos)
      if not isinstance(cal, str):
        cal = tuple(TupleCalibration(c) if not callable(cal) else c for c in cal)
      cacheable = (cal == "NonNaNMask" or all(isinstance(c, TupleCalibration) for c in cal))
      if cacheable:
        with contextlib.suppress(KeyError):
          return self.data[(mod, channelNoOrNos, cal)]
      if cal == "NonNaNMask":
        returnV = np.any([self.getLazy(mod, ch, "NonNaNMask") for ch in channelNoOrNos], axis=0)
      else: # Plain addback data is requested
        returnV = np.full(self.getLazy(mod, channelNoOrNos[0], cal[0]).shape, np.nan)
        returnV[self.getLazy(mod, channelNoOrNos, "NonNaNMask")] = 0
        for ch, c in zip(channelNoOrNos, cal):
          nonNaNMask = self.getLazy(mod, ch, "NonNaNMask")
          returnV[nonNaNMask] += self.getLazy(mod, ch, c)[nonNaNMask]
    if cacheable:
      self.data[(mod, channelNoOrNos, cal)] = returnV
    return returnV


@dataclasses.dataclass
class ExportSetting:
  "BACKLOG." # Meant for automatic constrcution of histograms (mutliple! Raw, Cal, Time, and beam gated! With multiple Binnings!)
  # masterSetting: ExportSetting = None # Redefine getattr to get from master
  module: str
  channels: Sequence[ChannelNo]
  addbackAutoGroup: dataclasses.InitVar[Optional[int]] = None # If not None addbackChannelsDict is constructed automatically by grouping consecutive channels together in groups of length addbackAutoGroup, grouping-remainder channels are ignored
  addbackChannelsDict: dict[int, tuple[ChannelNo, ...]] = dataclasses.field(default_factory=dict)
  amplitudeIdentifier: str = "amplitude[16]"
  channelTimeIdentifier: str = "channel_time[16]"
  beamRFSourceChannel: ChannelNo = 0
  beamRFSourceModule: Optional[str] = None
  beamRFIdentifier: str = "trigger_time[2]"

  amplitudeRawDigitizerRange: int = 2**16 # 2**12 for scintillators
  amplitudeRawHistsRebinningFactors: Sequence[int] = (1, ) # Different for scintillators
  amplitudeCalHistsMaxE: float = 20e3
  amplitudeCalHistsBinWidths: Sequence[float] = (1, ) # Different for scintillators
  timeChannelRawDigitizerRange: int = 2**16 # 2**14 for scintillators
  timeBeamRFRawDigitizerRange: Optional[int] = None # 2**14 for scintillators
  timeRawHistsRebinningFactors: Sequence[float] = (1, )
  timeChannelCalHistsMaxT: Optional[float] = None # Different for scintillators
  timeBeamRFCalHistsMaxT: Optional[float] = None # Different for scintillators
  timeCalHistsBinWidths: Sequence[float] = (1, ) # Different for scintillators
  timeSignalToRFCalHistsRange: Optional[tuple[float, float]] = None # Different for scintillators
  amplitudeSignalToRFTimeGateIntervalsDict: Union[tuple[float, float], dict[ChannelNo, tuple[float, float]]] = dataclasses.field(default_factory=dict) # Different for every detector

  modulePropertyJoiner: str = "/"
  histFillThreads: Optional[int] = None

  @property
  def amplitudeModElem(self) -> MvmeModuleElement:
    "BACKLOG."
    return self.module + self.modulePropertyJoiner + self.amplitudeIdentifier

  @property
  def channelTimeModElem(self) -> MvmeModuleElement:
    "BACKLOG."
    return self.module + self.modulePropertyJoiner + self.channelTimeIdentifier

  @property
  def beamRFModElem(self) -> MvmeModuleElement:
    "BACKLOG."
    return self.beamRFSourceModule + self.modulePropertyJoiner + self.beamRFIdentifier

  def __post_init__(self, addbackAutoGroup) -> None:
    "Called by __init__() generated by dataclasses.dataclass."
    self.channels = tuple(self.channels)
    if addbackAutoGroup is not None:
      self.addbackChannelsDict = {ch0 // addbackAutoGroup + 1: self.channels[ch0:ch0 + addbackAutoGroup] for ch0 in range(0, len(self.channels) - addbackAutoGroup + 1, addbackAutoGroup)}
    else:
      self.addbackChannelsDict = {k: tuple(v) for k, v, in self.addbackChannelsDict.items()}
    if self.beamRFSourceModule is None:
      self.beamRFSourceModule = self.module
    if self.timeBeamRFRawDigitizerRange is None:
      self.timeBeamRFRawDigitizerRange = self.timeChannelRawDigitizerRange
    if self.timeChannelCalHistsMaxT is None:
      self.timeChannelCalHistsMaxT = self.timeChannelRawDigitizerRange * ScaleCalibration.timeBinWidth
    if self.timeBeamRFCalHistsMaxT is None:
      self.timeBeamRFCalHistsMaxT = self.timeBeamRFRawDigitizerRange * ScaleCalibration.timeBinWidth
    if self.timeSignalToRFCalHistsRange is None:
      self.timeSignalToRFCalHistsRange = (-self.timeBeamRFCalHistsMaxT, self.timeChannelCalHistsMaxT)
    if isinstance(self.amplitudeSignalToRFTimeGateIntervalsDict, tuple):
      self.amplitudeSignalToRFTimeGateIntervalsDict = {ch: self.amplitudeSignalToRFTimeGateIntervalsDict for ch in self.channels}


@dataclasses.dataclass
class Config:
  "Container for mvmeRoot2Spec configuration."
  rootFilePath: str
  outDir: Optional[str] = None
  outPrefix: Optional[str] = None
  inCalFilePath: Optional[str] = None
  inCalPrefix: Optional[str] = None
  ignoreNoCalibrationFoundError: bool = True
  outCalFilePath: Optional[str] = None
  verbose: bool = False
  printProgress: bool = True
  buildRaw: bool = True
  buildCal: bool = True
  buildAddback: bool = True
  buildRawChannelTime: bool = True
  buildCalChannelTime: bool = True
  buildRawBeamRF: bool = True
  buildCalBeamRF: bool = True
  buildSignalToRF: bool = True
  buildRawSignalToRFTimeGated: bool = True
  buildCalSignalToRFTimeGated: bool = True
  disableExport: bool = False
  outcalOverwrite: bool = False
  uprootIterateStepSize: Union[str, int] = "800MB"
  uprootThreads: Optional[int] = min(16, os.cpu_count())
  histFillThreads: Optional[int] = min(8, os.cpu_count())
  maxEntriesToProcess: Optional[int] = None
  fractionOfEntriesToProcess: float = 1
  progressFractionThresholdStepSize: float = 0.05
  progressTimeThresholdStepSizeInMinutes: float = 0 if sys.stdout.isatty() else 3
  printF: updateablePrint = updateablePrint() # Must have keyword arg "updatable" in addition to usual print kw args
  mvmeModuleElements: Optional[Sequence[MvmeModuleElement]] = None
  mvmeModuleElementRenames: Optional[dict[MvmeModuleElement, MvmeModuleElement]] = None
  exportSettings: Sequence[ExportSetting] = None
  exportSettingsInitF: dataclasses.InitVar[Callable[[Config], Sequence[ExportSetting]]] = lambda cfg: [
    ExportSetting(module="clovers_up", channels=range(16), addbackAutoGroup=4),
    ExportSetting(module="clovers_down", channels=range(8), addbackAutoGroup=4),
    ExportSetting(module="scintillators", channels=range(14), amplitudeRawDigitizerRange=2**12, timeChannelRawDigitizerRange=2**14, amplitudeIdentifier="integration_long[16]", amplitudeRawHistsRebinningFactors=(1, ), amplitudeCalHistsBinWidths=(5, ), timeCalHistsBinWidths=(0.25, )),
    ExportSetting(module="zero_degree" if cfg.rootFilePathToRunNo() <= 715 else "clovers_down" if cfg.rootFilePathToRunNo() <= 744 else "zero_degree" if cfg.rootFilePathToRunNo() <= 781 else "clovers_sum", channels=(15, ), timeCalHistsBinWidths=(5, ) if 715 < cfg.rootFilePathToRunNo() <= 744 else (), timeChannelRawDigitizerRange=2**13),
    ExportSetting(module="zero_degree" if cfg.rootFilePathToRunNo() <= 781 else "clovers_sum", channels=(1, 2, 4, 5, 6, 8) if cfg.rootFilePathToRunNo() > 715 else (), timeCalHistsBinWidths=(), timeChannelRawDigitizerRange=2**13),
  ]

  def __post_init__(self, exportSettingsInitF) -> None:
    "Called by __init__() generated by dataclasses.dataclass. Here used to infer default values from other fields if necessary."
    if self.outPrefix is None:
      self.outPrefix = self.rootFilePathToPrefix()
    if self.inCalPrefix is None:
      self.inCalPrefix = self.outPrefix
    if self.outCalFilePath is None:
      self.outCalFilePath = f"{self.outPrefix}_{SORTED_STR}.{CALLST_EXT}"
    if self.outDir is None:
      self.outDir = f"{self.outPrefix}_{SORTED_STR}"
    if self.maxEntriesToProcess is not None and not isinstance(self.maxEntriesToProcess, int):
      if isinstance(self.maxEntriesToProcess, float) and self.maxEntriesToProcess.is_integer():
        self.maxEntriesToProcess = int(self.maxEntriesToProcess)
      else:
        raise ExplicitCheckFailedError("maxEntriesToProcess must be an integer!")
    if self.exportSettings is None:
      self.exportSettings = exportSettingsInitF(self)
    if self.mvmeModuleElements is None:
      self.mvmeModuleElements = set(mod for exportSetting in self.exportSettings for mod in (exportSetting.amplitudeModElem, exportSetting.channelTimeModElem, exportSetting.beamRFModElem))

  def rootFilePathToRunNo(self) -> int:
    "BACKLOG."
    match = re.search("mvmelst_(\d+)_raw", os.path.basename(self.rootFilePath))
    if match:
      return int(match.group(1))
    raise ExplicitCheckFailedError(f"Failed to extract RunNo from rootFilePath '{self.rootFilePath}'!")

  def rootFilePathToPrefix(self) -> str:
    "Turns a path to a (ROOT) file into a prefix, striping/converting only known components and leaving the rest as is."
    filePath = os.path.basename(self.rootFilePath)
    if filePath.startswith("mvmelst_"):
      filePath = RUN_STR + filePath[8:]
    if "." in filePath:
      filePath = filePath.rpartition(".")[0]
    if filePath.endswith("_raw"):
      filePath = filePath[:-4]
    return filePath

  def buildSimpleHistsGeneric(self, calibDictOrFalse: Union[Literal[False], CalibDict], bSpecConstructor: Callable[..., BinningSpec] = BinningSpec.constructEdgeAlignedBinning, exStBinWidthsAttr: str = "amplitudeRawHistsRebinningFactors", exStHistsMaxAttr: str = "amplitudeRawDigitizerRange", exStModElemAttr: str = "amplitudeModElem", exStChAttr: str = "channels") -> list[Hist1D]:
    "BACKLOG."
    hists = []
    for exSt in self.exportSettings:
      channels = getattr(exSt, exStChAttr)
      for chNo in (channels, ) if isinstance(channels, int) else channels:
        for binWidth in getattr(exSt, exStBinWidthsAttr):
          bSpec = bSpecConstructor(getattr(exSt, exStHistsMaxAttr), binWidth, exceptionOnRounding=False)
          with contextlib.suppress(NoCalibrationFoundError) if self.ignoreNoCalibrationFoundError else contextlib.nullcontext():
            hists.append(Hist1D.constructSimpleHist1D(self.outPrefix, getattr(exSt, exStModElemAttr), chNo, bSpec, calibDictOrFalse, fillThreads=self.histFillThreads))
    return hists

  def buildSimpleHists(self, calibDictOrFalse: Union[Literal[False], CalibDict]) -> list[Hist1D]:
    "BACKLOG."
    if calibDictOrFalse is False:
      return self.buildSimpleHistsGeneric(calibDictOrFalse, BinningSpec.constructEdgeAlignedBinning, "amplitudeRawHistsRebinningFactors", "amplitudeRawDigitizerRange", "amplitudeModElem", "channels")
    else:
      return self.buildSimpleHistsGeneric(calibDictOrFalse, BinningSpec.constructCenterAlignedBinning, "amplitudeCalHistsBinWidths", "amplitudeCalHistsMaxE", "amplitudeModElem", "channels")

  def buildSimpleChannelTimeHists(self, calibDictOrFalse: Union[Literal[False], CalibDict]) -> list[Hist1D]:
    "BACKLOG."
    if calibDictOrFalse is False:
      return self.buildSimpleHistsGeneric(calibDictOrFalse, BinningSpec.constructEdgeAlignedBinning, "timeRawHistsRebinningFactors", "timeChannelRawDigitizerRange", "channelTimeModElem", "channels")
    else:
      return self.buildSimpleHistsGeneric(calibDictOrFalse, BinningSpec.constructCenterAlignedBinning, "timeCalHistsBinWidths", "timeChannelCalHistsMaxT", "channelTimeModElem", "channels")

  def buildSimpleBeamRFHists(self, calibDictOrFalse: Union[Literal[False], CalibDict]) -> list[Hist1D]:
    "BACKLOG."
    if calibDictOrFalse is False:
      return self.buildSimpleHistsGeneric(calibDictOrFalse, BinningSpec.constructEdgeAlignedBinning, "timeRawHistsRebinningFactors", "timeBeamRFRawDigitizerRange", "beamRFModElem", "beamRFSourceChannel")
    else:
      return self.buildSimpleHistsGeneric(calibDictOrFalse, BinningSpec.constructCenterAlignedBinning, "timeCalHistsBinWidths", "timeBeamRFCalHistsMaxT", "beamRFModElem", "beamRFSourceChannel")

  def buildAddbackHists(self, calibDict: CalibDict) -> list[Hist1D]:
    "BACKLOG."
    hists = []
    for exSt in self.exportSettings:
      for addbackNo, chNos in exSt.addbackChannelsDict.items():
        for binWidth in exSt.amplitudeCalHistsBinWidths:
          bSpec = BinningSpec.constructCenterAlignedBinning(exSt.amplitudeCalHistsMaxE, binWidth, exceptionOnRounding=False)
          with contextlib.suppress(NoCalibrationFoundError) if self.ignoreNoCalibrationFoundError else contextlib.nullcontext():
            hists.append(Hist1D.constructAddbackHist1D(self.outPrefix, exSt.amplitudeModElem, addbackNo, chNos, bSpec, calibDict, fillThreads=self.histFillThreads))
    return hists

  def buildSignalToRFTimeHists(self, calibDict: CalibDict) -> list[Hist1D]:
    "BACKLOG."
    hists = []
    for exSt in self.exportSettings:
      for chNo in exSt.channels:
        for binWidth in exSt.timeCalHistsBinWidths:
          bSpec = BinningSpec.constructCenterAlignedBinning(exSt.timeSignalToRFCalHistsRange[1], binWidth, exSt.timeSignalToRFCalHistsRange[0], exceptionOnRounding=False)
          hists.append(Hist1D.constructSignalToRFTimeHist1D(self.outPrefix, exSt.channelTimeModElem, chNo, bSpec, calibDict, modBeamRF=exSt.beamRFModElem, channelNoBeamRF=exSt.beamRFSourceChannel, fillThreads=self.histFillThreads))
    return hists

  def buildSignalToRFTimeGatedHists(self, calibDict: Union[Literal[False], CalibDict], raw: bool = True) -> list[Hist1D]:
    "BACKLOG."
    hists = []
    for exSt in self.exportSettings:
      for chNo in exSt.channels:
        gate = exSt.amplitudeSignalToRFTimeGateIntervalsDict.get(chNo, None)
        if gate is not None:
          for binWidth in exSt.amplitudeRawHistsRebinningFactors if raw else exSt.amplitudeCalHistsBinWidths:
            if raw:
              bSpec = BinningSpec.constructEdgeAlignedBinning(exSt.amplitudeRawDigitizerRange, binWidth, exceptionOnRounding=False)
            else:
              bSpec = BinningSpec.constructCenterAlignedBinning(exSt.amplitudeCalHistsMaxE, binWidth, exceptionOnRounding=False)
            with contextlib.suppress(NoCalibrationFoundError) if self.ignoreNoCalibrationFoundError else contextlib.nullcontext():
              hists.append(Hist1D.constructSignalToRFTimeGatedHist1D(self.outPrefix, exSt.amplitudeModElem, chNo, gate, bSpec, False if raw else calibDict, calibDict, False, exSt.beamRFModElem, exSt.beamRFSourceChannel, exSt.channelTimeModElem, fillThreads=self.histFillThreads))
              hists.append(Hist1D.constructSignalToRFTimeGatedHist1D(self.outPrefix, exSt.amplitudeModElem, chNo, gate, bSpec, False if raw else calibDict, calibDict, True, exSt.beamRFModElem, exSt.beamRFSourceChannel, exSt.channelTimeModElem, fillThreads=self.histFillThreads))
    return hists

  def buildAllHists(self, calibDict: CalibDict) -> list[Hist1D]:
    "BACKLOG."
    hists = []
    if self.buildRaw:
      hists += self.buildSimpleHists(False)
    if self.buildCal:
      hists += self.buildSimpleHists(calibDict)
    if self.buildRawChannelTime:
      hists += self.buildSimpleChannelTimeHists(False)
    if self.buildCalChannelTime:
      hists += self.buildSimpleChannelTimeHists(calibDict)
    if self.buildRawBeamRF:
      hists += self.buildSimpleBeamRFHists(False)
    if self.buildCalBeamRF:
      hists += self.buildSimpleBeamRFHists(calibDict)
    if self.buildAddback:
      hists += self.buildAddbackHists(calibDict)
    if self.buildSignalToRF:
      hists += self.buildSignalToRFTimeHists(calibDict)
    if self.buildRawSignalToRFTimeGated:
      hists += self.buildSignalToRFTimeGatedHists(calibDict, True)
    if self.buildCalSignalToRFTimeGated:
      hists += self.buildSignalToRFTimeGatedHists(calibDict, False)
    return hists


@dataclasses.dataclass
class DataAccumulatorBase(abc.ABC):
  "BACKLOG"
  name: str
  detNoFormat: ClassVar[str] = "02"
  timeCalibrations: ClassVar[dict[str, Calibration]] = {prop: ScaleCalibration(ScaleCalibration.timeBinWidth) for prop in (ExportSetting.beamRFIdentifier, ExportSetting.channelTimeIdentifier)}

  @abc.abstractmethod
  def processModuleDictBatch(self, data: LazyCachedAugmentedModuleBatch) -> None:
    "BACKLOG."
    ...

  @abc.abstractmethod
  def export(self, outDir: str, calOutputFile: Optional[TextIO] = None, printF: Optional[updateablePrint] = updateablePrint()):
    "BACKLOG."
    ...

  @classmethod
  def createHistNamePart(cls, mod: MvmeModuleElement, label: str = DET_STR, channelNo: ChannelNo = 0, channelNoOffset: int = 1, propAppendix="") -> str:
    "BACKLOG."
    mod, _, prop = mod.partition(ExportSetting.modulePropertyJoiner)
    mod = {"scintillators": "Sci", "clovers_up": "CUp", "clovers_down": "CDown", "zero_degree": "FanInFanOut", "clovers_sum": "FanInFanOut"}.get(mod, mod)
    if prop in (ExportSetting.amplitudeIdentifier, "integration_long[16]"):
      prop = "E"
    if prop == ExportSetting.channelTimeIdentifier:
      prop = "Time"
    if prop == ExportSetting.beamRFIdentifier and channelNo == ExportSetting.beamRFSourceChannel:
      return f'{mod}_RF'
    if prop == "module_timestamp":
      return f'{mod}_ModTime'
    if channelNo == 15 and mod in ("CDown", "FanInFanOut"):
      return f'ZeroDegree_{prop}{propAppendix}'
    return f'{mod}_{prop}{propAppendix}_{label}_{channelNo+channelNoOffset:{cls.detNoFormat}}'

  @classmethod
  def findCalib(cls, mod: MvmeModuleElement, channelNo: ChannelNo, calibDict: CalibDict) -> Calibration:
    "BACKLOG."
    cal = calibDict.get(cls.createHistNamePart(mod, DET_STR, channelNo), None)
    if cal is not None:
      if not (isinstance(cal, tuple) or callable(cal)):
        raise ExplicitCheckFailedError("Calib not callable or poly coeffs BACKLOG")
      return cal
    for prop, cal in cls.timeCalibrations.items():
      if mod.endswith(prop):
        if not (isinstance(cal, tuple) or callable(cal)):
          raise ExplicitCheckFailedError("Calib not callable or poly coeffs BACKLOG")
        return cal
    raise NoCalibrationFoundError(f'No calib for {cls.createHistNamePart(mod, DET_STR, channelNo)} BACKLOG')


@dataclasses.dataclass
class DataAccumulator(DataAccumulatorBase):
  "BACKLOG"
  name: str
  dataProcessor: Callable[[DataAccumulator, LazyCachedAugmentedModuleBatch], None]

  def processModuleDictBatch(self, data: LazyCachedAugmentedModuleBatch) -> None:
    "BACKLOG."
    self.dataProcessor(self, data)

  def export(self, outDir: str, calOutputFile: Optional[TextIO] = None, printF: Optional[updateablePrint] = updateablePrint()):
    "BACKLOG."
    pass


@dataclasses.dataclass
class Hist1D(DataAccumulatorBase):
  "BACKLOG."
  name: str
  binningSpec: BinningSpec
  dataProcessor: Callable[[LazyCachedAugmentedModuleBatch], np.ndarray]
  fillThreads: Optional[int] = Config.histFillThreads
  histFlowBinsEnabled: bool = False
  h: hist.Hist = dataclasses.field(init=False)

  def __post_init__(self) -> None:
    "Called by __init__() generated by dataclasses.dataclass. Here used to initialize underlying hist.Hist object using other fields."
    self.h = hist.Hist.new.Regular(**self.binningSpec.asHistAxisKW(), flow=self.histFlowBinsEnabled).AtomicInt64()
    if self.fillThreads is None:
      self.fillThreads = type(self).fillThreads

  def processModuleDictBatch(self, data: LazyCachedAugmentedModuleBatch) -> None:
    "BACKLOG."
    self.h.fill(self.dataProcessor(data), threads=self.fillThreads)

  def export(self, outDir: str, calOutputFile: Optional[TextIO] = None, printF: Optional[updateablePrint] = updateablePrint()):
    "BACKLOG."
    if printF is not None:
      printF("Writing", self.name, "...", updatable=True)
    outFilePath = os.path.join(outDir, f"{self.name}.{FILE_EXT}")
    # np.savetxt(outFilePath, self.h.counts(), fmt="%d", newline="\n")
    with open(outFilePath, "w", newline="\n") as f: # np.savetxt does not allow to force LF lineending on Windows
      f.writelines(str(elem) + "\n" for elem in self.h.counts())
    if calOutputFile is not None:
      calOutputFile.write(f"{self.name}.{FILE_EXT}: {self.binningSpec.getHDTVCalibrationStr()}\n")

  @classmethod
  def createHistName(cls, prefix: str, mod: MvmeModuleElement, label: str, channelNo: ChannelNo, cal: bool, binWidth: float, channelNoOffset: int = 1, propAppendix="") -> str:
    "BACKLOG."
    return f'{prefix}_{cls.createHistNamePart(mod, label, channelNo, channelNoOffset, propAppendix)}_{CAL_STR if cal else RAW_STR}_{BINNING_STR}{binWidth}'

  @classmethod
  def constructSimpleHist1D(cls, prefix: str, mod: MvmeModuleElement, channelNo: ChannelNo, bSpec: BinningSpec, cal: Union[Literal[False], CalibDict, Calibration] = False, fillThreads: Optional[int] = None) -> Hist1D:
    "BACKLOG."
    name = cls.createHistName(prefix, mod, DET_STR, channelNo, cal is not False, bSpec.binWidth)
    if isinstance(cal, dict):
      cal = cls.findCalib(mod, channelNo, cal)
    return cls(name, bSpec, lambda data: data.getLazy(mod, channelNo, cal)[data.getLazy(mod, channelNo, "NonNaNMask")], fillThreads)

  @classmethod
  def constructAddbackHist1D(cls, prefix: str, mod: MvmeModuleElement, addbackNo: int, channelNos: tuple[ChannelNo, ...], bSpec: BinningSpec, cals: Union[CalibDict, tuple[Calibration, ...]], fillThreads: Optional[int] = None) -> Hist1D:
    "BACKLOG."
    name = cls.createHistName(prefix, mod, ADDBACK_STR, addbackNo, True, bSpec.binWidth, channelNoOffset=0)
    if isinstance(cals, dict):
      cals = tuple(cls.findCalib(mod, channelNo, cals) for channelNo in channelNos)
    if len(cals) != len(channelNos):
      raise ExplicitCheckFailedError("BACKLOG")
    return cls(name, bSpec, lambda data: data.getLazy(mod, channelNos, cals)[data.getLazy(mod, channelNos, "NonNaNMask")], fillThreads)

  @classmethod
  def constructDifferenceHist1D(cls, prefix: str, mod1: MvmeModuleElement, channelNo1: ChannelNo, mod2: MvmeModuleElement, channelNo2: ChannelNo, bSpec: BinningSpec, cals: Union[Literal[False], CalibDict, Tuple[Calibration, Calibration]] = False, fillThreads: Optional[int] = None) -> Hist1D:
    "BACKLOG."
    name = f'{prefix}_{cls.createHistNamePart(mod1, DET_STR, channelNo1)}_-_{cls.createHistNamePart(mod2, DET_STR, channelNo2)}_{CAL_STR if cals is not False else RAW_STR}_{BINNING_STR}{bSpec.binWidth}'
    if isinstance(cals, dict):
      cals = (cls.findCalib(mod1, channelNo1, cals), cls.findCalib(mod2, channelNo2, cals))
    elif cals is False:
      cals = (False, False)
    # Note that the same NonNaNMask is used on purpose for both arrays, otherwise the dimensions wouldn't match. One could also AND it with the other mask, but this probably takes longer than any potential speedup!
    return cls(name, bSpec, lambda data: data.getLazy(mod1, channelNo1, cals[0])[data.getLazy(mod1, channelNo1, "NonNaNMask")] - data.getLazy(mod2, channelNo2, cals[1])[data.getLazy(mod1, channelNo1, "NonNaNMask")], fillThreads)

  @classmethod
  def constructSignalToRFTimeHist1D(cls, prefix: str, mod: MvmeModuleElement, channelNo: ChannelNo, bSpec: BinningSpec, cals: Union[Literal[False], CalibDict, Tuple[Calibration, Calibration]] = {}, modBeamRF: MvmeModuleElement = None, channelNoBeamRF: ChannelNo = ExportSetting.beamRFSourceChannel, fillThreads: Optional[int] = None) -> Hist1D:
    "BACKLOG."
    if modBeamRF is None:
      modBeamRF = f"{mod.partition(ExportSetting.modulePropertyJoiner)[0]}{ExportSetting.modulePropertyJoiner}{ExportSetting.beamRFIdentifier}"
    hist = cls.constructDifferenceHist1D(prefix, mod, channelNo, modBeamRF, channelNoBeamRF, bSpec, cals, fillThreads)
    hist.name = cls.createHistName(prefix, mod, DET_STR, channelNo, cals is not False, bSpec.binWidth, propAppendix=TO_RF_APPENDIX_STR)
    return hist

  @classmethod
  def constructSignalToRFTimeGatedHist1D(cls, prefix: str, mod: MvmeModuleElement, channelNo: ChannelNo, gateInterval: tuple[float, float], bSpec: BinningSpec, eCal: Union[Literal[False], CalibDict, Calibration] = False, tCals: Union[CalibDict, Tuple[Calibration, Calibration]] = {}, invertGate: bool = False, modBeamRF: MvmeModuleElement = None, channelNoBeamRF: ChannelNo = ExportSetting.beamRFSourceChannel, modTime: MvmeModuleElement = None, channelNoT: ChannelNo = None, gatePeriod: float = 179, fillThreads: Optional[int] = None) -> Hist1D:
    "BACKLOG."
    if modBeamRF is None:
      modBeamRF = f"{mod.partition(ExportSetting.modulePropertyJoiner)[0]}{ExportSetting.modulePropertyJoiner}{ExportSetting.beamRFIdentifier}"
    if modTime is None:
      modTime = f"{mod.partition(ExportSetting.modulePropertyJoiner)[0]}{ExportSetting.modulePropertyJoiner}{ExportSetting.channelTimeIdentifier}"
    if channelNoT is None:
      channelNoT = channelNo
    if isinstance(eCal, dict):
      eCal = cls.findCalib(mod, channelNo, eCal)
    if isinstance(tCals, dict):
      tCals = (cls.findCalib(modTime, channelNoT, tCals), cls.findCalib(modBeamRF, channelNoBeamRF, tCals))
    name = cls.createHistName(prefix, mod, DET_STR, channelNo, eCal is not False, bSpec.binWidth, propAppendix=ON_BEAM_GATED_APPENDIX_STR if not invertGate else OFF_BEAM_GATED_APPENDIX_STR)
    lowerGate, upperGate = gateInterval
    if lowerGate >= upperGate:
      raise ExplicitCheckFailedError("BACKLOG.")
    lowerGate, upperGate = lowerGate % gatePeriod, upperGate % gatePeriod
    if lowerGate > upperGate:
      invertGate = not invertGate
      lowerGate, upperGate = upperGate, lowerGate

    def dataProcessor(data: LazyCachedAugmentedModuleBatch) -> np.ndarray: #
      timeToRFData = (data.getLazy(modTime, channelNoT, tCals[0])[data.getLazy(mod, channelNo, "NonNaNMask")] - data.getLazy(modBeamRF, channelNoBeamRF, tCals[1])[data.getLazy(mod, channelNo, "NonNaNMask")]) % gatePeriod
      if not invertGate: # Through this if any events with NaNs in the time data will always be discarded, also it's faster than inverting the whole bool array
        return (data.getLazy(mod, channelNo, eCal)[data.getLazy(mod, channelNo, "NonNaNMask")])[lowerGate <= timeToRFData & timeToRFData < upperGate]
      return (data.getLazy(mod, channelNo, eCal)[data.getLazy(mod, channelNo, "NonNaNMask")])[lowerGate > timeToRFData | timeToRFData >= upperGate]

    return cls(name, bSpec, dataProcessor, fillThreads)


@dataclasses.dataclass
class HistND(DataAccumulatorBase):
  "BACKLOG."
  name: str
  binningSpecs: Sequence[BinningSpec]
  dataProcessor: Callable[[LazyCachedAugmentedModuleBatch], Iterable[np.ndarray]]
  fillThreads: Optional[int] = Config.histFillThreads
  histFlowBinsEnabled: bool = False
  dim: int = dataclasses.field(init=False)
  h: hist.Hist = dataclasses.field(init=False)

  def __post_init__(self) -> None:
    "Called by __init__() generated by dataclasses.dataclass. Here used to initialize underlying hist.Hist object using other fields."
    self.dim = len(self.binningSpecs)
    h = hist.Hist.new
    for binningSpec in self.binningSpecs:
      h = h.Regular(**binningSpec.asHistAxisKW(), flow=self.histFlowBinsEnabled)
    self.h = h.AtomicInt64()
    if self.fillThreads is None:
      self.fillThreads = type(self).fillThreads

  def processModuleDictBatch(self, data: LazyCachedAugmentedModuleBatch) -> None:
    "BACKLOG."
    self.h.fill(*self.dataProcessor(data), threads=self.fillThreads)

  def export(self, outDir: str, calOutputFile: Optional[TextIO] = None, printF: Optional[updateablePrint] = updateablePrint()):
    "NOT IMPLEMENTED YET."
    # BACKLOG Implement this?

  @classmethod
  def createHistName(cls, prefix: str, mods: Iterable[MvmeModuleElement], labels: Iterable[str], channelNos: Iterable[ChannelNo], cals: Iterable[bool], binWidths: Iterable[float]) -> str:
    "BACKLOG."
    return prefix + "_VS_".join(f'{cls.createHistNamePart(mod, label, channelNo)}_{CAL_STR if cal else RAW_STR}_{BINNING_STR}{binWidth}' for mod, label, channelNo, cal, binWidth in zip(mods, labels, channelNos, cals, binWidths))

  __createHistName = createHistName # Use name mangling to get a private copy of the original function (used by constructSimpleHistND) in case a subclass overwrites it (e.g. Hist2D does that)

  @classmethod
  def constructSimpleHistND(cls, prefix: str, mods: Sequence[MvmeModuleElement], channelNos: Sequence[ChannelNo], bSpecs: Sequence[BinningSpec], cals: Union[Literal[False], CalibDict, Sequence[Union[Literal[False], CalibDict, Calibration]]] = False, fillThreads: Optional[int] = None) -> HistND:
    "BACKLOG."
    if cals is False or isinstance(cals, dict):
      cals = itertools.repeat(cals)
    cals = [cls.findCalib(mod, channelNo, cal) if isinstance(cal, dict) else cal for mod, channelNo, cal in zip(mods, channelNos, cals)]
    name = cls.__createHistName(prefix, mods, itertools.repeat(DET_STR), channelNos, cals, (bs.binWidth for bs in bSpecs))
    return cls(name, bSpecs, lambda data: tuple(data.getLazy(mod, channelNo, cal) for mod, channelNo, cal in zip(mods, channelNos, cals)), fillThreads)


@dataclasses.dataclass
class Hist2D(HistND):
  "BACKLOG."
  name: str
  dataProcessor: Callable[[LazyCachedAugmentedModuleBatch], Tuple[np.ndarray, np.ndarray]]
  dim: int = dataclasses.field(init=False, repr=False)

  def __init__(self, name: str, binningSpecsXandY: tuple[BinningSpec, BinningSpec], dataProcessor: Callable[[LazyCachedAugmentedModuleBatch], Tuple[np.ndarray, np.ndarray]], **histND_kw_args) -> None:
    "BACKLOG."
    if len(binningSpecsXandY) != 2:
      raise ExplicitCheckFailedError("BACKLOG.")
    return super().__init__(name, binningSpecsXandY, dataProcessor, **histND_kw_args)

  @property
  def binningSpecX(self) -> BinningSpec:
    return self.binningSpecs[0]

  @property
  def binningSpecY(self) -> BinningSpec:
    return self.binningSpecs[1]

  @classmethod
  def createHistName(cls, prefix: str, modX: MvmeModuleElement, labelX: str, channelNoX: ChannelNo, calX: bool, binWidthX: float, modY: MvmeModuleElement, labelY: str, channelNoY: ChannelNo, calY: bool, binWidthY: float) -> str:
    "BACKLOG."
    return super().createHistName(prefix, (modX, modY), (labelX, labelY), (channelNoX, channelNoY), (calX, calY), (binWidthX, binWidthY))

  @classmethod
  def constructSimpleHist2D(cls, prefix: str, modX: MvmeModuleElement, channelNoX: ChannelNo, bSpecX: BinningSpec, calX: Union[None, CalibDict, Calibration], modY: MvmeModuleElement, channelNoY: ChannelNo, bSpecY: BinningSpec, calY: Union[None, CalibDict, Calibration]) -> Hist2D:
    "BACKLOG."
    return super().constructSimpleHistND(prefix, (modX, modY), (channelNoX, channelNoY), (bSpecX, bSpecY), (calX, calY))


@dataclasses.dataclass
class Sorter:
  "Instances process mvme ROOT files to spectra and store the (intermediate) results."
  cfg: Config
  accs: list[DataAccumulatorBase] = dataclasses.field(default_factory=list)
  calibDict: CalibDict = dataclasses.field(default_factory=dict, init=False)

  def __post_init__(self) -> None:
    "Called by __init__() generated by dataclasses.dataclass. Here used to automatically parse the calibration file if one was specified in self.cfg."
    if self.cfg.inCalFilePath is not None:
      self.parseCal()

  def findAcc(self, name: str) -> Union[DataAccumulatorBase, tuple[DataAccumulatorBase, ...]]:
    "BACKLOG."
    matches = tuple(h for h in self.accs if name in h.name)
    if len(matches) == 1:
      return matches[0]
    return matches

  def parseCal(self: Sorter) -> None:
    "Parse calibration file specified in self.cfg and store relevant found calibrations in self.calibDict."
    if self.cfg.inCalFilePath is None or not os.path.isfile(self.cfg.inCalFilePath):
      raise ExplicitCheckFailedError(f"Supplied INCALIB '{self.cfg.inCalFilePath}' is no valid file!")
    calFilenameRegex = re.compile(f"{self.cfg.inCalPrefix}_(.+)_{RAW_STR}_{BINNING_STR}(\\d+)\\.{FILE_EXT}:?\s")
    with open(self.cfg.inCalFilePath) as f:
      for line in f:
        regexMatch = calFilenameRegex.match(line) # Also filters out un-splitable blank lines
        if regexMatch:
          cal = np.polynomial.Polynomial([float(x) for x in line.split()[1:]])
          # HDTV uses centered bins and exports the calibration matching this convention
          # The MVME DAQ ROOT exporter however randomizes the integer channel values by adding a [0,1) uniform distributed random number, i.e. assuming a integer-edge binning
          # Furthermore the raw data could have been binned with a binsize!=1 which hdtv does not know
          # To correct for this difference in schemes, create the composition of first shifting the raw channels to the hdtv
          # coresponding channel values (back by 0.5 and shrinked by 1/rawRebinningFactor used) and then applying the HDTV calibration
          calRebinningFactor = int(regexMatch.group(2))
          cal = cal(np.polynomial.Polynomial([-0.5, 1 / calRebinningFactor]))
          self.calibDict[regexMatch.group(1)] = TupleCalibration(cal.coef) # Store as TupleCalibration as they are hashable, i.e., useable as dict keys

  def constructHistsFromCfg(self: Sorter) -> None:
    "BACKLOG."
    self.accs = self.cfg.buildAllHists(self.calibDict)

  @contextlib.contextmanager
  def openRoot(self: Sorter) -> None:
    "Contextmanager (for with statements) to open ROOTFILE BACKLOG"
    if not os.path.isfile(self.cfg.rootFilePath):
      raise ExplicitCheckFailedError(f"Supplied ROOTFILE '{self.cfg.rootFilePath}' is no valid file!")
    uprootExecutor = uproot.ThreadPoolExecutor(num_workers=self.cfg.uprootThreads)
    with uproot.open(self.cfg.rootFilePath, num_workers=self.cfg.uprootThreads, decompression_executor=uprootExecutor, interpretation_executor=uprootExecutor) as rootFile:
      yield rootFile

  def processModuleDictBatch(self: Sorter, moduleDictBatch: MvmeDataBatch) -> None:
    "BACKLOG."
    lazyCachedAugmentedModuleBatch = LazyCachedAugmentedModuleBatch(moduleDictBatch, self.calibDict)
    for h in self.accs:
      h.processModuleDictBatch(lazyCachedAugmentedModuleBatch)

  def iterateRoot(self: Sorter) -> None:
    "Iterate over a mvme root file, handing each data batch to self.processModuleDictBatch."
    with self.openRoot() as rootFile:
      ev0: uproot.TBranch = rootFile['event0'] # Should actually be a TTree?
      totalEntries = ev0.num_entries
      processedEntries = 0
      entriesToProcess = totalEntries * self.cfg.fractionOfEntriesToProcess if self.cfg.maxEntriesToProcess is None or totalEntries * self.cfg.fractionOfEntriesToProcess < self.cfg.maxEntriesToProcess else self.cfg.maxEntriesToProcess
      self.cfg.printF(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f'- {totalEntries:.2e} entries{f" on file of which {entriesToProcess:.2e} entries are" if entriesToProcess != totalEntries else ""} to process.')
      uprootExecutor = uproot.ThreadPoolExecutor(num_workers=self.cfg.uprootThreads)
      if self.cfg.printProgress:
        startTime = datetime.datetime.now()
        nextProgressFraction = 0
        nextProgressTime = startTime
      moduleDictBatch: MvmeDataBatch
      for moduleDictBatch in iterateAhead(ev0.iterate(tuple(self.cfg.mvmeModuleElements), library="np", decompression_executor=uprootExecutor, interpretation_executor=uprootExecutor, step_size=self.cfg.uprootIterateStepSize)):
        if self.cfg.mvmeModuleElementRenames is not None:
          moduleDictBatch = {self.cfg.mvmeModuleElementRenames.get(mod, mod): batch for mod, batch in moduleDictBatch.items()}
        if self.cfg.fractionOfEntriesToProcess != 1:
          moduleDictBatch = {mod: batch[:math.ceil(self.cfg.fractionOfEntriesToProcess * len(batch))] for mod, batch in moduleDictBatch.items()}
        processedEntries += next(iter(moduleDictBatch.values())).shape[0]
        if self.cfg.maxEntriesToProcess is not None and processedEntries >= self.cfg.maxEntriesToProcess:
          moduleDictBatch = {mod: batch[:len(batch) - (processedEntries - self.cfg.maxEntriesToProcess)] for mod, batch in moduleDictBatch.items()}
          self.cfg.printF(f"Breaking BACKLOG")
          self.processModuleDictBatch(moduleDictBatch)
          break
        self.processModuleDictBatch(moduleDictBatch)
        if self.cfg.printProgress:
          now = datetime.datetime.now()
          if processedEntries / entriesToProcess >= nextProgressFraction or now >= nextProgressTime or processedEntries == entriesToProcess:
            nextProgressFraction = round(processedEntries / entriesToProcess / self.cfg.progressFractionThresholdStepSize) * self.cfg.progressFractionThresholdStepSize + self.cfg.progressFractionThresholdStepSize
            nextProgressTime = now + datetime.timedelta(minutes=self.cfg.progressTimeThresholdStepSizeInMinutes)
            remainingSeconds = int((entriesToProcess / processedEntries - 1) * (now - startTime).total_seconds())
            self.cfg.printF(now.strftime("%Y-%m-%d %H:%M:%S"), f"- Processed {processedEntries:.2e} entries so far ({processedEntries/totalEntries:7.2%}) - ETA: {remainingSeconds//(60*60)}:{(remainingSeconds//60)%60:02}:{remainingSeconds%60:02} - Mean processing speed: {processedEntries/(now - startTime).total_seconds():.2e} entries/s", updatable=True)
      self.cfg.printF.makeLastLineUnUpdatable() # Make updateable line un-updateable (if any), i.e. leaving info about speed intact, even if any updatable prints follow

  def exportSpectra(self: Sorter):
    "Export all histograms and a hdtv calibration file."
    if self.cfg.disableExport:
      self.cfg.printF("Skipping export due to active config setting 'disableExport'.")
      return
    if len(self.accs) == 0:
      self.cfg.printF("There were no spectra created! You might want to check your settings...?")
      return
    os.makedirs(self.cfg.outDir, exist_ok=True)
    outCalFullFilePath = os.path.join(self.cfg.outDir, self.cfg.outCalFilePath)
    if self.cfg.verbose:
      self.cfg.printF(f"Writing calibrations to {outCalFullFilePath}.")
    with open(outCalFullFilePath, "w" if self.cfg.outcalOverwrite else "a", newline="\n") as calOutputFile:
      for h in self.accs:
        h.export(self.cfg.outDir, calOutputFile, self.cfg.printF if self.cfg.verbose else None)

  def runSorting(self) -> None:
    "BACKLOG."
    self.constructHistsFromCfg()
    self.iterateRoot()
    self.exportSpectra()


def mvmeRoot2Spec(*cfg_args, **cfg_kw_args) -> Sorter:
  "mvmeRoot2Spec main function, processes an mvme ROOT file to spectra and exports the result, returning the Sorter object."
  if len(cfg_args) == 1 and len(cfg_kw_args) == 0 and isinstance(cfg_args[0], Config):
    cfg = cfg_args[0]
  else:
    cfg = Config(*cfg_args, **cfg_kw_args)
  srt = Sorter(cfg)
  srt.runSorting()
  return srt


def mvmeRoot2SpecCLI(argv: Sequence[str]) -> Sorter:
  "CLI interface to mvmeRoot2Spec, call mvmeRoot2SpecCLI(['-h']) to print the usage. Returns the Sorter object."

  # Parse command line arguments in an nice way with automatic usage message etc
  programName = os.path.basename(sys.argv[0]) if __name__ == "__main__" else __name__
  argparser = argparse.ArgumentParser(prog=programName, description=__doc__)

  # yapf: disable
  argparser.add_argument("rootFilePath", metavar="ROOTFILE", help="path to the ROOT file to process")
  argparser.add_argument("--incal", dest="inCalFilePath", metavar="INCALIB", default=None, type=str, help=f"path to a HDTV format energy calibration list. Energy calibrated spectra can be generated by {programName} for a channel if an energy calibration of the raw spectrum of the channel is found in this calibration file. I.e. just energy calibrate raw spectra exported by {programName} with HDTV, save the calibration list file and supply the calibration list file with this option to {programName} to be able to generate energy calibrated and addbacked spectra.")
  argparser.add_argument("--outprefix", dest="outPrefix", metavar="OUTPREFIX", default=None, type=str, help="prefix to the filenames of the output files. Default is a prefix based on the input ROOT file's name.")
  argparser.add_argument("--outcal", dest="outCalFilePath", metavar="OUTCALIB", default=None, type=str, help="path to write the HDTV format energy calibration list for the output spectra. If a relative path is used, it is taken relative to the output directory. Default is a file with its name based on OUTPREFIX and located in the output directory.")
  argparser.add_argument("--outdir", dest="outDir", metavar="OUTDIR", default=None, type=str, help="The directory path to write the output files to. Default is a directory with its name based on the input ROOT file's name and located in the current working directory.")
  argparser.add_argument("-v", "--verbose", dest="verbose", action="store_true", help="Use to enable verbose output")
  argparser.add_argument("-p", "--noprogress", dest="printProgress", action="store_false", help="Use to disable progress info output")
  argparser.add_argument("--noraw", dest="buildRaw", action="store_false", help="Use to disable output of raw spectra")
  argparser.add_argument("--nocal", dest="buildCal", action="store_false", help="Use to disable output of calibrated spectra (not affecting addbacked spectra)")
  argparser.add_argument("--noaddback", dest="buildAddback", action="store_false", help="Use to disable output of addbacked spectra")
  argparser.add_argument("--outcaloverwrite", dest="outcalOverwrite", action="store_true", help="Use to overwrite a potentially existing file, when writing the output energy calibration list file. If not used, writing appends to an existing file.")
  # yapf: enable

  cliArgs = argparser.parse_args(argv)
  startTime = datetime.datetime.now()
  print(startTime.strftime("%Y-%m-%d %H:%M:%S"), f"- Executing commandline '{programName}' '" + "' '".join(map(str, argv)) + "'")

  try:
    srt = mvmeRoot2Spec(Config(**vars(cliArgs)))
  except ExplicitCheckFailedError as e:
    # Catch any explicit/manual execution aborts and exit with an error message but without printing the full stack traceback.
    # This allows for a nice CLI interface with simple, concise error messages for simple/expected errors, while allowing to
    # properly handle such errors as exceptions when interfacing with this module's functions directly via other python code.
    sys.exit(f"ERROR: {e}")

  elapsedSeconds = int((datetime.datetime.now() - startTime).total_seconds())
  print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "-", programName, f"has finished, took {elapsedSeconds//(60*60)}:{(elapsedSeconds//60)%60:02}:{elapsedSeconds%60:02}")
  if not sys.stdout.isatty():
    print(80 * "#")
  return srt


if __name__ == "__main__":
  mvmeRoot2SpecCLI(sys.argv[1:])

# import cProfile
# import pstats
# if __name__ == "__main__":
#   with cProfile.Profile() as pr:
#     mvmeRoot2SpecCLI(sys.argv[1:])
#   stats = pstats.Stats(pr)
#   stats.sort_stats(pstats.SortKey.TIME)
#   stats.dump_stats(filename="stats.prof")
