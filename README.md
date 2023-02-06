# Repetuition
>IPPT Static Station rep counter using Computer Vision, implemented for AISG Student Challenge 2022

## Dependencies
The python environment required for **Repetuition** is stored in `environment.yml`.
### To create this environment locally:
1. Clone the **Repetuition** repository locally and install the latest version of Anaconda (https://www.anaconda.com/products/distribution).
2. Open the terminal or an Anaconda Prompt at local repository location and run this:
```
conda env create -f environment.yml
```

## First time **Repetuition** user:
1. Run this command in terminal at reporsitory location to run a one-time calibration mode:
```
peekingduck run
```
2. **Repetuition** runs in calibration mode to calibrate to your full range of motion and body. Ensure you perform all push-up reps in perfect form in this mode.
3. Once done with calibration, exit program by pressing `Q` on the keyboard.
4. Then, run the following command again to enter IPPT test mode:
```
peekingduck run
```
5. Begin doin push-ups when ready and the GUI will show the time left and number of reps counted. Reps stop counting at the end of 60 seconds.
6. Use the IPPT test mode at the same location as many times as required.
7. Delete `distance.txt` and `angles.txt` before changing user or location for IPPT test mode, restart from step 1.

### Recommendations for the best experience
- Wear tight fitting clothing (PT kit) and do not wear loose fitting jewellery or watches so that the PoseNet model can accurately identify body joints.
- Once done with calibration, do not move the device and/or camera from its position. Start the IPPT test mode in the same location.