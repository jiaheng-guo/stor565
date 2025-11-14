# STOR 565 Group 13 Final Project

This repository contains the code and paper for the final project of STOR 565 at UNC Chapel Hill. The project focuses on benchmarking classical and modern boosting algorithms on various datasets spanning regression and classification tasks. The final paper can be found at `./paper/outputs/main.pdf`

To reproduce the results, please run the following command in the root directory:

```bash
python3 -m venv team13
source team13/bin/activate
pip install -r requirements.txt
python "./src/main.py"
```