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

"""TODO Module docstring"""

import os
import sys
import re
import math
import datetime
import argparse
from functools import lru_cache as functoolsLRUCache
import numpy as np
import uproot

def mvmeRoot2Spec(
  rootFilePath,
  inCalFilePath = None,
  outCalFilePath = None,
  outDir = None,
  rawHistRebinningFactors = None,
  calHistBinningSpecs = None,
  verbose = False,
  exportRaw = True,
  exportCal = True,
  exportAddback = True,
  outcalOverwrite = False,
):

  # The given ROOT file must exist
  if not os.path.isfile(rootFilePath) :
      exit("ERROR: Supplied ROOTFILE '" + rootFilePath + "' is no valid file!")

  startTime = datetime.datetime.now()
  print(startTime.strftime("%Y-%m-%d %H:%M:%S"), "- Executing commandline", "'"+"' '".join(map(str, sys.argv))+"'")

  # Helper to get nice filename prefix from rootFilePath
  @functoolsLRUCache(maxsize=None)
  def rootFilePathToPrefix(filePath):
    filePath = os.path.basename(filePath)
    if filePath.startswith("mvmelst_") :
      filePath = "run"+filePath[8:]
    if "." in filePath :
      filePath = filePath.rpartition(".")[0]
    if filePath.endswith("_raw") :
      filePath = filePath[:-4]
    return filePath

  # Default values for cliArgs
  if outCalFilePath is None :
    outCalFilePath = rootFilePathToPrefix(rootFilePath)+"_sorted.callst"
  if outDir is None :
    outDir = rootFilePathToPrefix(rootFilePath)+"_sorted"
  if rawHistRebinningFactors is None :
    rawHistRebinningFactors = [1]
  if calHistBinningSpecs is None :
    calHistBinningSpecs = [[1., 20e3]]

  os.makedirs(outDir, exist_ok=True)


  # Hardcoded settings
  mvmeModulesChannelsDict = {
    'clovers_up/amplitude[16]': range(16),
    'clovers_down/amplitude[16]': [0, 1, 2, 3, 4, 5, 6, 7],
    'clovers_sum/amplitude[16]': [15],
    'scintillators/integration_long[16]': range(14),
  }
  addbackChannelDict = {
    'clovers_up/amplitude[16]': {1: [0, 1, 2, 3], 2: [4, 5, 6, 7], 3: [8, 9, 10, 11], 4: [12, 13, 14, 15]},
    'clovers_down/amplitude[16]':   {1: [0, 1, 2, 3], 2: [4, 5, 6, 7], 3: [8, 9, 10, 11], 4: [12, 13, 14, 15]},
  }

  # Global variables and constants
  calsDict = dict()
  hists = dict()
  digitizerBins = 2**16
  calFilenameRegex = re.compile(fr"({rootFilePathToPrefix(rootFilePath)}_.+)_raw_b(\d+).txt:?")

  rawHistBinnings = [{
    "binWidth": rebinningFactor,
    "lowerEdge": 0,
    "nBins":     math.ceil(digitizerBins/rebinningFactor),
    "upperEdge": math.ceil(digitizerBins/rebinningFactor)*rebinningFactor
    } for rebinningFactor in rawHistRebinningFactors]

  calHistBinnings = [{
    "binWidth": binWidth,
    "lowerEdge": -binWidth/2,
    "nBins":     math.ceil((upperEdge+binWidth/2)/binWidth),
    "upperEdge": -binWidth/2+math.ceil((upperEdge+binWidth/2)/binWidth)*binWidth
    } for binWidth, upperEdge in calHistBinningSpecs]


  # Parse calibration file and store relevant found calibrations in a dict
  if inCalFilePath is not None:
    with open(inCalFilePath) as f:
      for line in f:
        specName, *valStrs = line.split()
        regexMatch = calFilenameRegex.fullmatch(specName)
        if regexMatch :
          cal = np.polynomial.Polynomial(np.array([float(x) for x in valStrs]))
          # HDTV uses centered bins and exports the calibration matching this convention
          # The MVME DAQ ROOT exporter however randomizes the integer channel values by adding a [0,1) uniform distributed random number, i.e. assuming a integer-edge binning
          # Furthermore the raw data could have been binned with a binsize!=1 which hdtv does not know
          # To correct for this difference in scheme, create the composition of first shifting the raw channels to the hdtv
          # coresponding channel values (back by 0.5 and shrinked by 1/rawRebinningFactor used) and then applying the HDTV calibration
          calRebinningFactor = int(regexMatch.group(2))
          cal = cal(np.polynomial.Polynomial([-0.5, 1/calRebinningFactor]))
          calsDict[regexMatch.group(1)] = cal


  # Helper function to create the prefix of histogram filenames
  @functoolsLRUCache(maxsize=None)
  def getHistFilenamePrefix(rootFilename, module, detNo) :
    return rootFilePathToPrefix(rootFilename)+"_"+module.partition("/")[0]+"_"+str(detNo)


  # Helper function to sort events into histograms, accumulate histograms from blocks of data and initialize histograms
  def accumulateHist(histNamePrefix, data, histBinning, calState) :
    histKey = histNamePrefix+"_"+calState+"_b"+str(histBinning["binWidth"])+".txt"
    if histKey not in hists:
      hists[histKey] = {"calState": calState, "binning": histBinning, "cts": np.zeros(histBinning["nBins"])}
    hists[histKey]["cts"] += np.histogram(data, histBinning["nBins"], (histBinning["lowerEdge"], histBinning["upperEdge"]))[0]
    # hists[histKey] = hists.get(histKey, np.zeros(histBinning["nBins"])) + np.histogram(data, histBinning["nBins"], (histBinning["lowerEdge"], histBinning["upperEdge"]))[0]


  # Helper function to process a data block of a detector
  def processDataBlock(dataID, data) :
    histNamePrefix = getHistFilenamePrefix(*dataID)
    if exportRaw:
      for rawHistBinning in rawHistBinnings:
        accumulateHist(histNamePrefix, data, rawHistBinning, "raw")
    if exportCal and histNamePrefix in calsDict :
      data = calsDict[histNamePrefix](data)
      for calHistBinning in calHistBinnings:
        accumulateHist(histNamePrefix, data, calHistBinning, "cal")


  # Helper function to process a data block of multiple channels for addback
  def processAddbackDataBlock(dataID, channels, data) :
    addbackData = np.zeros(data.shape[1])
    for channelNo, channelData in zip(channels, data) :
      calName = getHistFilenamePrefix(*dataID[:2], f"Det_{channelNo+1:02}")
      if calName not in calsDict :
        return
      # To optimize performance one could try reusing the calibrated data from the processDataBlock call, but beware that
      # at the moment the processDataBlock has its NaNs dropped and dropping NaNs early speeds up the following computations noticeably...
      channelData = calsDict[calName](channelData)
      addbackData += np.nan_to_num(channelData)
    addbackData = addbackData[addbackData!=0]
    histNamePrefix = getHistFilenamePrefix(*dataID)
    for calHistBinning in calHistBinnings:
      accumulateHist(histNamePrefix, addbackData, calHistBinning, "cal")

  # Main loop over the data
  processedEvents = 0
  nextProgressPercentage = 0
  nextProgressTime = startTime
  uprootExecutor=uproot.ThreadPoolExecutor(num_workers=os.cpu_count())
  with uproot.open(rootFilePath, num_workers=os.cpu_count(), decompression_executor=uprootExecutor, interpretation_executor=uprootExecutor) as rootFile :
    totalEntries = rootFile['event0'].num_entries
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f"- {totalEntries:.2e} events to process.")
    for moduleDictBlock in rootFile['event0'].iterate(tuple(mvmeModulesChannelsDict), library="np", decompression_executor=uprootExecutor, interpretation_executor=uprootExecutor, step_size="50 MB") :
      for moduleName, moduleData in moduleDictBlock.items() :
        if exportRaw or exportCal:
          for channelNo in mvmeModulesChannelsDict[moduleName] :
            detData = moduleData[:,channelNo]
            detData = detData[~np.isnan(detData)]
            processDataBlock((rootFilePath, moduleName, f"Det_{channelNo+1:02}"), detData)
        if exportAddback and inCalFilePath is not None and moduleName in addbackChannelDict :
          for addbackNo, channels in addbackChannelDict[moduleName].items() :
            processAddbackDataBlock((rootFilePath, moduleName, f"Addback_{addbackNo}"), channels, moduleData.T[channels])
      if verbose:
        processedEvents += moduleData.shape[0]
        now = datetime.datetime.now()
        if True or processedEvents/totalEntries >= nextProgressPercentage or now >=  nextProgressTime:
          nextProgressPercentage = round(processedEvents/totalEntries/0.05) * 0.05 + 0.05
          nextProgressTime = now + datetime.timedelta(minutes=3)
          remainingSeconds = int((totalEntries/processedEvents - 1) * (now - startTime).total_seconds())
          print(now.strftime("%Y-%m-%d %H:%M:%S"), f"- Processed {processedEvents:.2e} events so far (≈{processedEvents/totalEntries:7.2%}) - ETA: {remainingSeconds//(60*60)}:{(remainingSeconds//60)%60:02}:{remainingSeconds%60:02} - Mean processing speed: {processedEvents/(now - startTime).total_seconds():.2e} events/s")

  # Export histograms and a hdtv calibration file
  if len(hists) > 0 :
    if verbose:
      print("Writing calibrations to", os.path.join(outDir, outCalFilePath))
    with open(os.path.join(outDir, outCalFilePath) , "w" if outcalOverwrite else "a") as calOutputFile :
      for histName, hist in hists.items() :
        if verbose:
          print("Writing",histName,"...")
        outfilename = os.path.join(outDir, histName)
        np.savetxt(outfilename, hist["cts"], fmt="%d")
        calOutputFile.write(histName + ": " + str(hist["binning"]["binWidth"]/2+hist["binning"]["lowerEdge"]) + "   " + str(hist["binning"]["binWidth"]) + "\n")
  else:
    print(now.strftime("%Y-%m-%d %H:%M:%S"), "- There were no spectra created! You might want to check your settings...?")

  elapsedSeconds = int((datetime.datetime.now()-startTime).total_seconds())
  print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"-", programName, f"has finished, took {elapsedSeconds//(60*60)}:{(elapsedSeconds//60)%60:02}:{elapsedSeconds%60:02}")
  if verbose:
    print(80*"#")



def mvmeRoot2SpecCLI(argv):
  def parseRawHistBinningSpecs(argStr) :
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

  def parseCalHistBinningSpecs(argStr) :
    exception = argparse.ArgumentTypeError("Binning specifications for energy calibrated spectra must be a comma-separated list of square-bracketed, comma-separated tuples of positive numbers!")
    try:
      match = re.fullmatch(r"\s*\[\s*(.*)\s*\]\s*", argStr)
      if not match:
        raise exception
      argStr = match.group(1)
      argList = re.split(r"\s*\]\s*,\s*\[\s*", argStr)
      argList = [re.split(r"\s*,\s*", tupleStr) for tupleStr in argList]
      argList = [[float(i) for i in tup] for tup in argList]
      if any(len(tup)!=2 or any(i <= 0 for i in tup) for tup in argList):
        raise exception
      return argList
    except:
      raise exception


  # Parse command line arguments in an nice way with automatic usage message etc
  programName = os.path.basename(sys.argv[0])
  argparser = argparse.ArgumentParser(description="Sort events of a ROOT file exported by the mvme_root_client from a Mesytech VME DAQ into spectra (histograms). Specifically aimed at the mvme DAQ used since 2021 at the High Intensity γ-ray Source (HIγS) facility, located at the Triangle Universities Nuclear Laboratory in Durham, NC, USA.")

  argparser.add_argument("rootFilePath", metavar="ROOTFILE", help="path to the ROOT file to process")
  argparser.add_argument("--incal", dest="inCalFilePath", metavar="INCALIB", default=None, type=str, help=f"path to a HDTV format energy calibration list. Energy calibrated spectra can be generated by {programName} for a channel if an energy calibration of the raw spectrum of the channel is found in this calibration file. I.e. just energy calibrate raw spectra exported by {programName} with HDTV, save the calibration list file and supply the calibration list file with this option to {programName} to be able to generate energy calibrated and addbacked spectra.")
  argparser.add_argument("--outcal", dest="outCalFilePath", metavar="OUTCALIB", default=None, type=str, help="path to write the HDTV format energy calibration list for the output spectra. If a relative path is used, it is taken relative to the output directory. Default is a file with its name based on the input ROOT file's name and located in the output directory.")
  argparser.add_argument("--outdir", dest="outDir", metavar="OUTDIR", default=None, type=str, help="The directory path to write the output files to. Default is a directory with its name based on the input ROOT file's name and located in the current working directory.")
  argparser.add_argument("--rawrebinningfactors", dest="rawHistRebinningFactors", metavar="FACTORLIST", default=None, type=parseRawHistBinningSpecs, help="A list of rebinning factors to be used for raw spectrum generation. For each factor N a correspondingly binned raw spectrum will be generated by merging N raw bins into one. The list needs to be a single string with the factors separated by commas and the factors need to be positive integers, e.g. \"1,4,8\". Use 1 for no rebinning. Default is 1.")
  argparser.add_argument("--calbinningspecs", dest="calHistBinningSpecs", metavar="BINNINGSPECS", default=None, type=parseCalHistBinningSpecs, help="A list of binning specification tuples to be used for calibrated spectrum generation. For each tuple [bw,emax] a calibrated spectrum with binwidth bw and maximum energy emax (rounded up to fit the binwidth) will be generated for each channel, for which an energy calibration was found. The list needs to be a single string with the tuples separated by commas, the tuples being surrounded by square-brackets and the tuples being two positive numbers, e.g. \"[1,10000],[10,18000]\". Default is [1,20000].")
  argparser.add_argument("-v","--verbose", dest="verbose", action="store_true", help="Use to enable verbose output")
  argparser.add_argument("--noraw", dest="exportRaw", action="store_false", help="Use to disable output of raw spectra")
  argparser.add_argument("--nocal", dest="exportCal", action="store_false", help="Use to disable output of calibrated spectra (not affecting addbacked spectra)")
  argparser.add_argument("--noaddback", dest="exportAddback", action="store_false", help="Use to disable output of addbacked spectra")
  argparser.add_argument("--outcaloverwrite", dest="outcalOverwrite", action="store_true", help="Use to overwrite a potentially existing file, when writing the output energy calibration list file. If not used, writing appends to an existing file.")

  cliArgs = argparser.parse_args(argv)
  mvmeRoot2Spec(**vars(cliArgs))


if __name__ == "__main__":
  mvmeRoot2SpecCLI(sys.argv[1:])