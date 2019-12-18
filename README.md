# stats_451_final_project

Run the following commands from the terminal on linux/mac

To setup:

```
python3 -m venv env
source env/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e .

```

After setup run:

```
python3 avocado/models/TrainTest.py
```

Depending on your operating system, running the above line may automatically
open the matplotlib plotting environment with the guassian process visualization.

If this is not the case, then the plot will be stored as a png file in /figures.
This folder has many image files, the most recently generated will be at the bottom.
