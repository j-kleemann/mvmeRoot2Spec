# mvmeRoot2Spec

[![License: GPL-3.0](https://img.shields.io/github/license/j-kleemann/mvmeRoot2Spec?color=blue)](LICENSE)

Sort events of a ROOT file exported by the mvme_root_client from a Mesytech VME DAQ into spectra (histograms).
Specifically aimed at the mvme DAQ used since 2021 at the High Intensity γ-ray Source (HIγS) facility, located at the Triangle Universities Nuclear Laboratory in Durham, NC, USA.

## Requirements

* Python>=3.6
* [numpy](https://numpy.org/)
* [uproot4](https://github.com/scikit-hep/uproot4)

## Usage

The `mvmeRoot2Spec.py` python script is provided to sort the events into spectra.
Execute

```bash
$ ./mvmeRoot2Spec.py --help
```
to print the `mvmeRoot2Spec.py` usage message listing all positional and optional command line arguments.

## License

Copyright © 2021

Jörn Kleemann `<jkleemann@ikp.tu-darmstadt.de>`

This code is distributed under the terms of the GNU General Public License, version 3 or later. See [LICENSE](LICENSE) for more information.

## Acknowledgements

We thank U. Friman-Gayer and O. Papst for valuable discussions.

This work has been funded by the German state of Hesse under the grant “Nuclear Photonics” within the LOEWE program.

J. Kleemann acknowledges support by the Helmholtz Graduate School for Hadron and Ion Research of the Helmholtz Association.
