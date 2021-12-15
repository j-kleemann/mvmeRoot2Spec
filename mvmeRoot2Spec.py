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

# TODO Implement 2D Hist Export

from __future__ import annotations
import abc
import sys
import os
import math
import datetime
import argparse
from typing import Any, Callable, ClassVar, Iterable, Iterator, Sequence, Literal, NewType, Optional, TextIO, Tuple, Union
import dataclasses
import contextlib
from itertools import repeat
import re
import concurrent.futures
import numpy as np
import uproot
import hist

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
  rawHistRebinningFactors: list[int] = (1, )
  calHistBinningSpecs: list[tuple[float, float]] = ((1, 20e3), )
  verbose: bool = False
  printProgress: bool = False
  exportRaw: bool = True
  exportCal: bool = True
  exportAddback: bool = True
  disableExport: bool = True
  outcalOverwrite: bool = False
  uprootIterateStepSize: Union[str, int] = "800MB"
  uprootThreads: Optional[int] = min(16, os.cpu_count())
  histFillThreads: Optional[int] = min(8, os.cpu_count())
  maxEntriesToProcess: int = -1
  fractionOfEntriesToProcess: float = 1
  progressFractionThresholdStepSize: float = 0.05
  progressTimeThresholdStepSizeInMinutes: float = 0 if sys.stdout.isatty() else 3
  printF: updateablePrint = updateablePrint() # Must have keyword arg "updatable" in addition to usual print kw args
  mvmeModules: Optional[Sequence[MvmeModuleElement]] = None
  mvmeModulesDigitizerRangeDict: dict[MvmeModuleElement, int] = dataclasses.field(default_factory=lambda: {
    'clovers_up/amplitude[16]': 2**16,
    'clovers_down/amplitude[16]': 2**16,
    'clovers_sum/amplitude[16]': 2**16,
    'scintillators/integration_long[16]': 2**12,
  })
  mvmeModulesChannelsDict: ChannelsDict = dataclasses.field(default_factory=lambda: {
    'clovers_up/amplitude[16]': range(16),
    'clovers_down/amplitude[16]': [0, 1, 2, 3, 4, 5, 6, 7],
    # 'clovers_sum/amplitude[16]': [15], # TODO Reinclude
    'scintillators/integration_long[16]': range(14),
  })
  addbackChannelsDict: dict[MvmeModuleElement, dict[int, tuple[ChannelNo, ...]]] = dataclasses.field(default_factory=lambda: {
    'clovers_up/amplitude[16]': {1: (0, 1, 2, 3), 2: (4, 5, 6, 7), 3: (8, 9, 10, 11), 4: (12, 13, 14, 15)},
    'clovers_down/amplitude[16]': {1: (0, 1, 2, 3), 2: (4, 5, 6, 7), 3: (8, 9, 10, 11), 4: (12, 13, 14, 15)},
  })

  def __post_init__(self) -> None:
    "Called by __init__() generated by dataclasses.dataclass. Here used to infer default values from other fields if necessary."
    if self.outPrefix is None:
      self.outPrefix = self.rootFilePathToPrefix(self.rootFilePath)
    if self.inCalPrefix is None:
      self.inCalPrefix = self.outPrefix
    if self.outCalFilePath is None:
      self.outCalFilePath = self.outPrefix + "_sorted.callst"
    if self.outDir is None:
      self.outDir = self.outPrefix + "_sorted"
    if self.mvmeModules is None:
      self.mvmeModules = tuple(self.mvmeModulesChannelsDict)
    if not isinstance(self.maxEntriesToProcess, int):
      if isinstance(self.maxEntriesToProcess, float) and self.maxEntriesToProcess.is_integer():
        self.maxEntriesToProcess = int(self.maxEntriesToProcess)
      else:
        raise ExplicitCheckFailedError("maxEntriesToProcess must be an integer!")

  @staticmethod
  def rootFilePathToPrefix(filePath: str) -> str:
    "Turns a path to a (ROOT) file into a prefix, striping/converting only known components and leaving the rest as is."
    filePath = os.path.basename(filePath)
    if filePath.startswith("mvmelst_"):
      filePath = "run" + filePath[8:]
    if "." in filePath:
      filePath = filePath.rpartition(".")[0]
    if filePath.endswith("_raw"):
      filePath = filePath[:-4]
    return filePath


@dataclasses.dataclass(frozen=True)
class BinningSpec():
  "Container for histogram binning specifications. Ensures compability of upperEdge to lowerEdge and binWidth and calculates corresponding nBins."
  lowerEdge: float
  upperEdge: float
  binWidth: float
  nBins: int = dataclasses.field(init=False)
  exceptionOnRoundUpperEdge: bool = dataclasses.field(default=True, repr=False, compare=False)

  @classmethod
  def constructZeroCenteredBinning(cls, upperEdge: float, binWidth: float, exceptionOnRoundUpperEdge: bool = True) -> BinningSpec:
    "Constructs a BinningSpec with the first bin centered around 0."
    return cls(-binWidth / 2, upperEdge, binWidth, exceptionOnRoundUpperEdge)

  @classmethod
  def constructZeroEdgedBinning(cls, upperEdge: float, binWidth: float, exceptionOnRoundUpperEdge: bool = True) -> BinningSpec:
    "Constructs a BinningSpec with lowerEdge=0."
    return cls(0, upperEdge, binWidth, exceptionOnRoundUpperEdge)

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
class DataAccumulatorBase(abc.ABC):
  "BACKLOG"
  name: str
  detNoFormat: ClassVar[str] = "02"
  timeCalibrations: ClassVar[dict[str, Calibration]] = {prop: ScaleCalibration(25 / 1024) for prop in ("trigger_time[2]", "channel_time[16]")} # 25 / 1024 was the used MDPP-16 TDC resolution, see also tdc_resolution in https://www.mesytec.com/products/datasheets/MDPP-16_SCP-RCP.pdf

  @abc.abstractmethod
  def processModuleDictBatch(self, data: LazyCachedAugmentedModuleBatch) -> None:
    "BACKLOG."
    ...

  @abc.abstractmethod
  def export(self, outDir: str, calOutputFile: Optional[TextIO] = None, printF: Optional[updateablePrint] = updateablePrint()):
    "BACKLOG."
    ...

  @classmethod
  def createHistNamePart(cls, mod: MvmeModuleElement, label: str = "Det", channelNo: ChannelNo = 0, channelNoOffset: int = 1) -> str:
    "BACKLOG."
    mod, _, prop = mod.partition("/")
    if prop == "trigger_time[2]" and ChannelNo == 0:
      return f'{mod}_RF'
    if prop == "module_timestamp":
      return f'{mod}_ModTime'
    if prop in ("amplitude[16]", "integration_long[16]"):
      prop = "E"
    if prop == "channel_time[16]":
      prop = "Time"
    return f'{mod}_{prop}_{label}_{channelNo+channelNoOffset:{cls.detNoFormat}}'

  @classmethod
  def findCalib(cls, mod: MvmeModuleElement, channelNo: ChannelNo, calibDict: CalibDict) -> Calibration:
    "BACKLOG."
    cal = calibDict.get(cls.createHistNamePart(mod, "Det", channelNo), None)
    if cal is not None:
      if not (isinstance(cal, tuple) or callable(cal)):
        raise ExplicitCheckFailedError("Calib not callable or poly coeffs BACKLOG")
      return cal
    for prop, cal in cls.timeCalibrations.items():
      if mod.endswith(prop):
        if not (isinstance(cal, tuple) or callable(cal)):
          raise ExplicitCheckFailedError("Calib not callable or poly coeffs BACKLOG")
        return cal
    raise NoCalibrationFoundError(f'No calib for {cls.createHistNamePart(mod, "Det", channelNo)} BACKLOG')


@dataclasses.dataclass
class DataAccumulator(DataAccumulatorBase):
  "BACKLOG"
  name: str
  dataProcessor: Callable[[DataAccumulatorBase, LazyCachedAugmentedModuleBatch], None]

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
  h: hist.Hist = dataclasses.field(init=False)

  def __post_init__(self) -> None:
    "Called by __init__() generated by dataclasses.dataclass. Here used to initialize underlying hist.Hist object using other fields."
    self.h = hist.Hist.new.Regular(**self.binningSpec.asHistAxisKW(), flow=False).AtomicInt64()

  def processModuleDictBatch(self, data: LazyCachedAugmentedModuleBatch) -> None:
    "BACKLOG."
    self.h.fill(self.dataProcessor(data), threads=self.fillThreads)

  def export(self, outDir: str, calOutputFile: Optional[TextIO] = None, printF: Optional[updateablePrint] = updateablePrint()):
    "BACKLOG."
    if printF is not None:
      printF("Writing", self.name, "...", updatable=True)
    outFilePath = os.path.join(outDir, self.name + ".txt")
    # np.savetxt(outFilePath, self.h.counts(), fmt="%d", newline="\n")
    with open(outFilePath, "w", newline="\n") as f: # np.savetxt does not allow to force LF lineending on Windows
      f.writelines(str(elem) + "\n" for elem in self.h.counts())
    if calOutputFile is not None:
      calOutputFile.write(f"{self.name}.txt: {self.binningSpec.getHDTVCalibrationStr()}\n")

  @classmethod
  def createHistName(cls, prefix: str, mod: MvmeModuleElement, label: str, channelNo: ChannelNo, cal: bool, binWidth: float, channelNoOffset: int = 1) -> str:
    "BACKLOG."
    return f'{prefix}_{cls.createHistNamePart(mod, label, channelNo, channelNoOffset)}_{"cal" if cal else "raw"}_b{binWidth}'

  @classmethod
  def constructSimpleHist1D(cls, prefix: str, mod: MvmeModuleElement, channelNo: ChannelNo, bSpec: BinningSpec, cal: Union[Literal[False], CalibDict, Calibration] = False) -> Hist1D:
    "BACKLOG."
    name = cls.createHistName(prefix, mod, "Det", channelNo, cal is not None, bSpec.binWidth)
    if isinstance(cal, dict):
      cal = cls.findCalib(mod, channelNo, cal)
    return cls(name, bSpec, lambda data: data.getLazy(mod, channelNo, cal)[data.getLazy(mod, channelNo, "NonNaNMask")])

  @classmethod
  def constructAddbackHist1D(cls, prefix: str, mod: MvmeModuleElement, addbackNo: int, channelNos: tuple[ChannelNo, ...], bSpec: BinningSpec, cals: Union[CalibDict, tuple[Calibration, ...]]) -> Hist1D:
    "BACKLOG."
    name = cls.createHistName(prefix, mod, "Addback", addbackNo, True, bSpec.binWidth, channelNoOffset=0)
    if isinstance(cals, dict):
      cals = tuple(cls.findCalib(mod, channelNo, cals) for channelNo in channelNos)
    if len(cals) != len(channelNos):
      raise ExplicitCheckFailedError("BACKLOG")
    return cls(name, bSpec, lambda data: data.getLazy(mod, channelNos, cals)[data.getLazy(mod, channelNos, "NonNaNMask")])


@dataclasses.dataclass
class HistND(DataAccumulatorBase):
  "BACKLOG."
  name: str
  binningSpecs: Sequence[BinningSpec]
  dataProcessor: Callable[[LazyCachedAugmentedModuleBatch], Iterable[np.ndarray]]
  fillThreads: Optional[int] = Config.histFillThreads
  dim: int = dataclasses.field(init=False)
  h: hist.Hist = dataclasses.field(init=False)

  def __post_init__(self) -> None:
    "Called by __init__() generated by dataclasses.dataclass. Here used to initialize underlying hist.Hist object using other fields."
    self.dim = len(self.binningSpecs)
    h = hist.Hist.new
    for binningSpec in self.binningSpecs:
      h = h.Regular(**binningSpec.asHistAxisKW(), flow=False)
    self.h = h.AtomicInt64()

  def processModuleDictBatch(self, data: LazyCachedAugmentedModuleBatch) -> None:
    "BACKLOG."
    self.h.fill(*self.dataProcessor(data), threads=self.fillThreads)

  def export(self, outDir: str, calOutputFile: Optional[TextIO] = None, printF: Optional[updateablePrint] = updateablePrint()):
    "BACKLOG."

  @classmethod
  def createHistName(cls, prefix: str, mods: Iterable[MvmeModuleElement], labels: Iterable[str], channelNos: Iterable[ChannelNo], cals: Iterable[bool], binWidths: Iterable[float]) -> str:
    "BACKLOG."
    return prefix + "_VS_".join(f'{cls.createHistNamePart(mod, label, channelNo)}_{"cal" if cal else "raw"}_b{binWidth}' for mod, label, channelNo, cal, binWidth in zip(mods, labels, channelNos, cals, binWidths))

  __createHistName = createHistName # Use name mangling to get a private copy of the original function (used by constructSimpleHistND) in case a subclass overwrites it (e.g. Hist2D does that)

  @classmethod
  def constructSimpleHistND(cls, prefix: str, mods: Sequence[MvmeModuleElement], channelNos: Sequence[ChannelNo], bSpecs: Sequence[BinningSpec], cals: Union[Literal[False], CalibDict, Sequence[Union[Literal[False], CalibDict, Calibration]]] = False) -> HistND:
    "BACKLOG."
    if cals is False or isinstance(cals, dict):
      cals = repeat(cals)
    cals = [cls.findCalib(mod, channelNo, cal) if isinstance(cal, dict) else cal for mod, channelNo, cal in zip(mods, channelNos, cals)]
    name = cls.__createHistName(prefix, mods, repeat("Dets"), channelNos, cals, (bs.binWidth for bs in bSpecs))
    return cls(name, bSpecs, lambda data: tuple(data.getLazy(mod, channelNo, cal) for mod, channelNo, cal in zip(mods, channelNos, cals)))


@dataclasses.dataclass
class Hist2D(HistND):
  "BACKLOG."
  name: str
  dataProcessor: Callable[[LazyCachedAugmentedModuleBatch], Tuple[np.ndarray, np.ndarray]]
  dim: int = dataclasses.field(init=False, repr=False)

  def __init__(self, name: str, binningSpecsXandY: tuple[BinningSpec, BinningSpec], dataProcessor: Callable[[LazyCachedAugmentedModuleBatch], Tuple[np.ndarray, np.ndarray]]) -> None:
    "BACKLOG."
    if len(binningSpecsXandY) != 2:
      raise ExplicitCheckFailedError("BACKLOG.")
    return super().__init__(name, binningSpecsXandY, dataProcessor)

  @property
  def binningSpecX(self) -> BinningSpec:
    return self.binningSpecs[0]

  @property
  def binningSpecY(self) -> BinningSpec:
    return self.binningSpecs[1]

  def export(self, outDir: str, calOutputFile: Optional[TextIO] = None, printF: Optional[updateablePrint] = updateablePrint()):
    "BACKLOG."

  @classmethod
  def createHistName(cls, prefix: str, modX: MvmeModuleElement, labelX: str, channelNoX: ChannelNo, calX: bool, binWidthX: float, modY: MvmeModuleElement, labelY: str, channelNoY: ChannelNo, calY: bool, binWidthY: float) -> str:
    "BACKLOG."
    return super().createHistName(prefix, (modX, modY), (labelX, labelY), (channelNoX, channelNoY), (calX, calY), (binWidthX, binWidthY))

  @classmethod
  def constructSimpleHist2D(cls, prefix: str, modX: MvmeModuleElement, channelNoX: ChannelNo, bSpecX: BinningSpec, calX: Union[None, CalibDict, Calibration], modY: MvmeModuleElement, channelNoY: ChannelNo, bSpecY: BinningSpec, calY: Union[None, CalibDict, Calibration]) -> Hist2D:
    "BACKLOG."
    return super().constructSimpleHistND(prefix, (modX, modY), (channelNoX, channelNoY), (bSpecX, bSpecY), (calX, calY))


# @dataclasses.dataclass
# class Hist2D(DataAccumulatorBase):
#   "BACKLOG."
#   name: str
#   binningSpecX: BinningSpec
#   binningSpecY: BinningSpec
#   dataProcessor: Callable[[LazyCachedAugmentedModuleBatch], Tuple[np.ndarray, np.ndarray]]
#   fillThreads: Optional[int] = Config.histFillThreads
#   h: hist.Hist = dataclasses.field(init=False)

#   def __post_init__(self) -> None:
#     "Called by __init__() generated by dataclasses.dataclass. Here used to initialize underlying hist.Hist object using other fields."
#     self.h = hist.Hist.new.Regular(**self.binningSpecX.asHistAxisKW(), flow=False).Regular(**self.binningSpecY.asHistAxisKW(), flow=False).AtomicInt64()

#   def processModuleDictBatch(self, data: LazyCachedAugmentedModuleBatch) -> None:
#     "BACKLOG."
#     self.h.fill(*self.dataProcessor(data), threads=self.fillThreads)

#   def export(self, outDir: str, calOutputFile: Optional[TextIO] = None, printF: Optional[updateablePrint] = updateablePrint()):
#     "BACKLOG."

#   @classmethod
#   def createHistName(cls, prefix: str, modX: MvmeModuleElement, labelX: str, channelNoX: ChannelNo, calX: bool, binWidthX: float, modY: MvmeModuleElement, labelY: str, channelNoY: ChannelNo, calY: bool, binWidthY: float) -> str:
#     "BACKLOG."
#     return f'{prefix}_{cls.createHistNamePart(modX, labelX, channelNoX)}_{"cal" if calX else "raw"}_b{binWidthX}_VS_{cls.createHistNamePart(modY, labelY, channelNoY)}_{"cal" if calY else "raw"}_b{binWidthY}'

#   @classmethod
#   def constructSimpleHist2D(cls, prefix: str, modX: MvmeModuleElement, channelNoX: ChannelNo, bSpecX: BinningSpec, calX: Union[Literal[False], CalibDict, Calibration], modY: MvmeModuleElement, channelNoY: ChannelNo, bSpecY: BinningSpec, calY: Union[Literal[False], CalibDict, Calibration]) -> Hist2D:
#     "BACKLOG."
#     name = cls.createHistName(prefix, modX, "Det", channelNoX, calX, bSpecX.binWidth, modY, "Det", channelNoY, calY, bSpecY.binWidth)
#     if isinstance(calX, dict):
#       calX = cls.findCalib(modX, channelNoX, calX)
#     if isinstance(calY, dict):
#       calY = cls.findCalib(modY, channelNoY, calY)
#     return cls(name, bSpecX, bSpecY, lambda data: (getLazy(modX, channelNoX, calX), getLazy(modY, channelNoY, calY)))


@dataclasses.dataclass
class Sorter:
  "Instances process mvme ROOT files to spectra and store the (intermediate) results."
  cfg: Config
  hists: list[DataAccumulatorBase] = dataclasses.field(default_factory=list)
  calibDict: CalibDict = dataclasses.field(default_factory=dict, init=False)

  # hists: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict, init=False)

  def parseCal(self: Sorter) -> None:
    "Parse calibration file specified in self.cfg and store relevant found calibrations in self.calibDict."
    if self.cfg.inCalFilePath is None or not os.path.isfile(self.cfg.inCalFilePath):
      raise ExplicitCheckFailedError(f"Supplied INCALIB '{self.cfg.inCalFilePath}' is no valid file!")
    calFilenameRegex = re.compile(f"{self.cfg.inCalPrefix}_(.+)_raw_b(\\d+).txt:?\s")
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

  def constructHistsFromCfg(self: Sorter) -> None: #TODO make this more tidy
    "BACKLOG."
    for mod, channelNos in self.cfg.mvmeModulesChannelsDict.items():
      for channelNo in channelNos:
        if self.cfg.exportRaw:
          for binWidth in self.cfg.rawHistRebinningFactors:
            bSpec = BinningSpec.constructZeroEdgedBinning(self.cfg.mvmeModulesDigitizerRangeDict[mod], binWidth, False)
            self.hists.append(Hist1D.constructSimpleHist1D(self.cfg.outPrefix, mod, channelNo, bSpec))
        if self.cfg.exportCal:
          for binWidth, eMax in self.cfg.calHistBinningSpecs:
            bSpec = BinningSpec.constructZeroCenteredBinning(eMax, binWidth, False)
            try:
              self.hists.append(Hist1D.constructSimpleHist1D(self.cfg.outPrefix, mod, channelNo, bSpec, self.calibDict))
            except NoCalibrationFoundError:
              if not self.cfg.ignoreNoCalibrationFoundError:
                raise
    if self.cfg.exportAddback:
      for mod, addbackChannelDict in self.cfg.addbackChannelsDict.items():
        for addbackNo, channelNos in addbackChannelDict.items():
          for binWidth, eMax in self.cfg.calHistBinningSpecs:
            bSpec = BinningSpec.constructZeroCenteredBinning(eMax, binWidth, False)
            try:
              self.hists.append(Hist1D.constructAddbackHist1D(self.cfg.outPrefix, mod, addbackNo, channelNos, bSpec, self.calibDict))
            except NoCalibrationFoundError:
              if not self.cfg.ignoreNoCalibrationFoundError:
                raise
    for acc in self.hists:
      if hasattr(acc, "fillThreads"):
        acc.fillThreads = self.cfg.histFillThreads

  def processModuleDictBatch(self: Sorter, moduleDictBatch: MvmeDataBatch) -> None:
    "BACKLOG."
    lazyCachedAugmentedModuleBatch = LazyCachedAugmentedModuleBatch(moduleDictBatch, self.calibDict)
    for h in self.hists:
      h.processModuleDictBatch(lazyCachedAugmentedModuleBatch)

  @contextlib.contextmanager
  def openRoot(self: Sorter) -> None:
    "Contextmanager (for with statements) to open ROOTFILE BACKLOG"
    if not os.path.isfile(self.cfg.rootFilePath):
      raise ExplicitCheckFailedError(f"Supplied ROOTFILE '{self.cfg.rootFilePath}' is no valid file!")
    uprootExecutor = uproot.ThreadPoolExecutor(num_workers=self.cfg.uprootThreads)
    with uproot.open(self.cfg.rootFilePath, num_workers=self.cfg.uprootThreads, decompression_executor=uprootExecutor, interpretation_executor=uprootExecutor) as rootFile:
      yield rootFile

  def iterateRoot(self: Sorter) -> None:
    "Iterate over a mvme root file, handing each data batch to self.processModuleDictBatch."
    with self.openRoot() as rootFile:
      ev0: uproot.TBranch = rootFile['event0'] # Should actually be a TTree?
      totalEntries = ev0.num_entries
      processedEntries = 0
      entriesToProcess = totalEntries * self.cfg.fractionOfEntriesToProcess if self.cfg.maxEntriesToProcess < 0 or totalEntries * self.cfg.fractionOfEntriesToProcess < self.cfg.maxEntriesToProcess else self.cfg.maxEntriesToProcess
      self.cfg.printF(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f'- {totalEntries:.2e} entries{f" on file of which {entriesToProcess:.2e} entries are" if entriesToProcess != totalEntries else ""} to process.')
      uprootExecutor = uproot.ThreadPoolExecutor(num_workers=self.cfg.uprootThreads)
      if self.cfg.printProgress:
        startTime = datetime.datetime.now()
        nextProgressFraction = 0
        nextProgressTime = startTime
      moduleDictBatch: MvmeDataBatch
      for moduleDictBatch in iterateAhead(ev0.iterate(tuple(self.cfg.mvmeModules), library="np", decompression_executor=uprootExecutor, interpretation_executor=uprootExecutor, step_size=self.cfg.uprootIterateStepSize)):
        # self.cfg.printF(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "start")
        if self.cfg.fractionOfEntriesToProcess != 1:
          moduleDictBatch = {mod: batch[:math.ceil(self.cfg.fractionOfEntriesToProcess * len(batch))] for mod, batch in moduleDictBatch.items()}
        processedEntries += next(iter(moduleDictBatch.values())).shape[0]
        if self.cfg.maxEntriesToProcess >= 0 and processedEntries >= self.cfg.maxEntriesToProcess:
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
        # self.cfg.printF(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "stop")
      self.cfg.printF.makeLastLineUnUpdatable() # Make updateable line un-updateable (if any), i.e. leaving info about speed intact, even if any updatable prints follow

  def exportSpectra(self: Sorter):
    "Export all histograms and a hdtv calibration file."
    if self.cfg.disableExport:
      if self.cfg.verbose:
        self.cfg.printF("Skipping export due to active config setting 'disableExport'.")
      return
    if len(self.hists) == 0:
      self.cfg.printF("There were no spectra created! You might want to check your settings...?")
      return
    os.makedirs(self.cfg.outDir, exist_ok=True)
    outCalFullFilePath = os.path.join(self.cfg.outDir, self.cfg.outCalFilePath)
    if self.cfg.verbose:
      self.cfg.printF(f"Writing calibrations to {outCalFullFilePath}.")
    with open(outCalFullFilePath, "w" if self.cfg.outcalOverwrite else "a", newline="\n") as calOutputFile:
      for h in self.hists:
        h.export(self.cfg.outDir, calOutputFile, self.cfg.printF if self.cfg.verbose else None)

  def runSorting(self) -> None:
    "BACKLOG."
    if self.cfg.inCalFilePath is not None:
      self.parseCal()
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

  def parseRawHistBinningSpecs(argStr: str) -> list[int]:
    "Helper function for argparse.ArgumentParser.add_argument to parse and verify the raw rebinning factors as a comma-separated list of positive integer numbers"
    exception = argparse.ArgumentTypeError("Rebinning factors for raw spectra must be comma-separated positive integer numbers!")
    try:
      if "." in argStr:
        raise exception
      argList = [int(s) for s in argStr.split(",")]
      if any(i <= 0 for i in argList):
        raise exception
      return argList
    except:
      raise exception

  def parseCalHistBinningSpecs(argStr: str) -> list[tuple[float, float]]:
    "Helper function for argparse.ArgumentParser.add_argument to parse and verify the binning specifications as a comma-separated list of square-bracketed, comma-separated tuples of positive numbers"
    exception = argparse.ArgumentTypeError("Binning specifications for energy calibrated spectra must be a comma-separated list of square-bracketed, comma-separated tuples of positive numbers!")
    try:
      match = re.fullmatch(r"\s*\[\s*(.*)\s*\]\s*", argStr)
      if not match:
        raise exception
      argStr = match.group(1)
      argList = re.split(r"\s*\]\s*,\s*\[\s*", argStr)
      argList = [re.split(r"\s*,\s*", tupleStr) for tupleStr in argList]
      argList = [[float(i) for i in tup] for tup in argList]
      if any(len(tup) != 2 or any(i <= 0 for i in tup) for tup in argList):
        raise exception
      return argList
    except:
      raise exception

  # Parse command line arguments in an nice way with automatic usage message etc
  programName = os.path.basename(sys.argv[0]) if __name__ == "__main__" else __name__
  argparser = argparse.ArgumentParser(prog=programName, description=__doc__)

  # yapf: disable
  argparser.add_argument("rootFilePath", metavar="ROOTFILE", help="path to the ROOT file to process")
  argparser.add_argument("--incal", dest="inCalFilePath", metavar="INCALIB", default=None, type=str, help=f"path to a HDTV format energy calibration list. Energy calibrated spectra can be generated by {programName} for a channel if an energy calibration of the raw spectrum of the channel is found in this calibration file. I.e. just energy calibrate raw spectra exported by {programName} with HDTV, save the calibration list file and supply the calibration list file with this option to {programName} to be able to generate energy calibrated and addbacked spectra.")
  argparser.add_argument("--outcal", dest="outCalFilePath", metavar="OUTCALIB", default=None, type=str, help="path to write the HDTV format energy calibration list for the output spectra. If a relative path is used, it is taken relative to the output directory. Default is a file with its name based on the input ROOT file's name and located in the output directory.")
  argparser.add_argument("--outdir", dest="outDir", metavar="OUTDIR", default=None, type=str, help="The directory path to write the output files to. Default is a directory with its name based on the input ROOT file's name and located in the current working directory.")
  argparser.add_argument("--rawrebinningfactors", dest="rawHistRebinningFactors", metavar="FACTORLIST", default=[1], type=parseRawHistBinningSpecs, help="A list of rebinning factors to be used for raw spectrum generation. For each factor N a correspondingly binned raw spectrum will be generated by merging N raw bins into one. The list needs to be a single string with the factors separated by commas and the factors need to be positive integers, e.g. \"1,4,8\". Use 1 for no rebinning. Default is 1.")
  argparser.add_argument("--calbinningspecs", dest="calHistBinningSpecs", metavar="BINNINGSPECS", default=[[1., 20e3]], type=parseCalHistBinningSpecs, help="A list of binning specification tuples to be used for calibrated spectrum generation. For each tuple [bw,emax] a calibrated spectrum with binwidth bw and maximum energy emax (rounded up to fit the binwidth) will be generated for each channel, for which an energy calibration was found. The list needs to be a single string with the tuples separated by commas, the tuples being surrounded by square-brackets and the tuples being two positive numbers, e.g. \"[1,10000],[10,18000]\". Default is [1,20000].")
  argparser.add_argument("-v", "--verbose", dest="verbose", action="store_true", help="Use to enable verbose output")
  argparser.add_argument("-p", "--progress", dest="printProgress", action="store_true", help="Use to enable progress info output")
  argparser.add_argument("--noraw", dest="exportRaw", action="store_false", help="Use to disable output of raw spectra")
  argparser.add_argument("--nocal", dest="exportCal", action="store_false", help="Use to disable output of calibrated spectra (not affecting addbacked spectra)")
  argparser.add_argument("--noaddback", dest="exportAddback", action="store_false", help="Use to disable output of addbacked spectra")
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
#   stats=pstats.Stats(pr)
#   stats.sort_stats(pstats.SortKey.TIME)
#   stats.dump_stats(filename="stats.prof")
