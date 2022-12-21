Aloha Renderer
==============

Data parallel (OptiX+MPI+IceT) renderer for simple structured volumetric and
surface models, accompanying our research paper:

S. Zellmann, I. Wald, J. Barbosa, S. Demirci, A. Sahistan, U. Gudukbay (2022):
Hybrid Image-/Data-Parallel Rendering Using Island Parallelism, in 2022 IEEE
12th Symposium on Large Data Analysis and Visualization (LDAV)

This is "Renderer 1" from the paper.

Apps: `waikikiRenderer` (for rendering). `chopSuey` is the partitioner; raw
data files and power-of-two partitionings work.

Building
--------

Usual CUDA + OptiX 7 control flow. IceT is a submodule, but you should build it
yourself and install it to some common location; linking against external IceT
made life a little easier when building against non-system MPI.

Misc.
-----

Islands mode: (`--num-islands` arg to the viewer app) allows you to use data
replication, as described in the paper. Each island holds a full copy of the
model and renderers 1/numIslands share of the screen.

The renderer has a CPU mode, for HPC systems that don't have GPU compute: cmake
variable `ALOHA_CPU`.

In [/viewer.cpp](viewer.cpp), the variable `bool runSlavesInWindow` might come
in handy for debugging.

Citation
--------

If this research is useful to you, please cite the following paper:
```
@INPROCEEDINGS{Zellmann:2022:Islands,
  author={Zellmann, Stefan and Wald, Ingo and Barbosa, Joao and Demirci, Serkan and Sahistan, Alper and G{\"u}d{\"u}kbay, U\u{g}ur},
  booktitle={2022 IEEE 12th Symposium on Large Data Analysis and Visualization (LDAV)},
  title={Hybrid Image-/Data-Parallel Rendering Using Island Parallelism},
  year={2022},
  volume={},
  number={},
  pages={1-10},
  doi={10.1109/LDAV57265.2022.9966396}}
```

Acknowledgments
---------------

This work was partially funded by the Deutsche Forschungsgemeinschaft (DFG,
German Research Foundation) under grant no. 456842964. We also thank the
Ministry of Culture and Science of the State of North Rhine-Westphalia for
supporting the work through the PROFILBILDUNG grant PROFILNRW-2020-038C.

