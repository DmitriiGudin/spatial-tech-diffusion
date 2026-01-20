# Prerequisites

Python 3.12


# Installation

1. Download the file at https://bit.ly/trackingthesun2025 and unzip the large csv-file to *data/raw/pv/*. It should be named *TTS_LBNL_public_file_29-Sep-2025_all.csv*, otherwise rename it to this.

2. Run
```
python raw_to_processed_solar_all.py
```


# Usage

We will use Florida here as an example.

*  *(Optional)* Before modeling the market, we need to choose the state to explore and approximate its geometry with a mesh. You can play around with mesh parameters *h_km* (mesh size) and *simplify_km* (boundary simplification size) and explore figures and diagnostics:
```
python run_mesh_diag.py --h_km 8 --simplify_km 24 --states CA
```
Typically *simplify_km* of 3 x *h_km* works well. The numbers to look out for are the number of triangles (a few thousand tends to give a very close approximation to the true PDE solution) and the number of obtuse triangles (ideally no more than a few, otherwise serious local numerical errors are possible).

* Make any necessary changes in *configs.py*. The default parameters were used in the dissertation. For Florida, the key in the file is *FL*.

* Run the MLE algorithm. To use 8 CPU processes:
```
python run_mle_parallel.py --n 8 --prefix fl_my_run --config FL
```
This will run 8 parallel processes each starting with random search, then multi-stage refinement of the best candidates. Once complete, the output csv-files will be saved to folders fl_my_run1, fl_my_run2, ..., fl_my_run8.

*
