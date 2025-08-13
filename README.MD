# Antibacklash project

- 3DModels -> 3D prints for the test rig
- antibacklashlib/antibacklashsys -> CDPStudio code
- Python -> all python scrips and .csv-files
    - data -> all .csv and .txt files from scripts, megalog2.csv and results_from_megalog2.txt the most recent. A lot of 'false positives' sometimes not correctly filtered out.
    - main.py -> the one to use

Kommentarer til antibacklash:
    Vårt testoppsett fungerer ofte dårlig ved lav speed. Neste steg som hadde vært et mulig steg for et mer reellt oppsett hadde vært joystick som input speed signal, og teste PD/PI regulering på posisjon med en FF på antall grader backlash i systemet. Vårt oppsett implementert som en P-regulator, uten FF.