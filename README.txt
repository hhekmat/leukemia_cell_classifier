INSTRUCTIONS ON HOW TO RUN:

1. Ensure you have all packages installed
        use pip install (or pip3 install) to get those you don't have
        WARNING: cv2 might have dependency issues with the most up-to-date
            version of NumPy... I had to downgrade my NumPy version to
            1.20.0 using 'pip install numpy==1.20.0' to get it to run!

2. Download the image dataset from kaggle at this link:
        https://www.kaggle.com/datasets/avk256/cnmc-leukemia
        For my main.py's path structure to work properly, you have to create
        a subfolder in my project folder titled 'images', along with two subfolders
        within the 'images' folder titled 'all' for malignant and 'hem' for normal.
        You can move all the data from the 'all' folders from the kaggle dataset 
        into your newly made 'all' subfolder, and do the same for the 'hem'
        data as well. (my apologies for the inconvenient file path structuring...
        gradescope kept crashing when I tried to upload my image folder directly,
        so I had to add this step to my instructions last minute as the deadline
        was approaching)
            
3. If you only wish to run certain part of script (e.g. PCA, 
    random forest, neural net) --> comment out function calls you don't want
    to run in the main() function. I recommend running PCA separately from either
    of the classifiers for efficiency's sake.

4. Run the script! 
        In VSCode, you can simply press the play button on the top right corner
        when you're in main.py. Alternatively, as long as you're in the
        directory for this project, you can simply run 'python main.py' in the
        terminal (or python3)
            My apologies that my code is definitely not the most time optimized - 
            thank you for your patience while it flattens all the images, trains 
            the models, and generates results!
            For the PCA aspect, I call PCA separately on just the malignant cells and 
            just the healthy ones. For each of these runs, it generates 2 matplotlib 
            visuals. You have to exit out of one for the next to pop up!
            For random forest and neural network results, check the print statements
            in the terminal!

Thanks so much :)