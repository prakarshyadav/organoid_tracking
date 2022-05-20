"""
Run as >>> python main.py PATH_TO_VIDEOS
PATH_TO_VIDEOS can be either relative or absolute

Requires: directories "data/" and "video_out/" for saving intermediate data

Import necessary libraries and scripts to run this program
"""
import numpy as np
import glob, sys
import support_fn


def angular_vel_calc(folder_name):
    """
    Function for parsing videos in a directory, loading them, calculating 
    angular velocity,and saving the data. The angular velocity data is saved 
    in the same directory as the video files
    
    Inputs
    -------
    folder_name (str) : A string to the PATH of the organoid videos
    
    Returns
    -------
    None.

    """
    folder = folder_name+'/'
    data_write = []
    for file_name in glob.glob(folder+"*.WMV"):
        print("Video file being analyzed",file_name)
        # Function for loading video, shape = (frames, height, width, channels)
        video_data = support_fn.vid_load(file_name)
        # Function for calculating angular velocity
        vel = support_fn.rot_vel_semi(video_data)
        # Output format for saving data
        data = [file_name, str(vel)]
        data_write.append(data)
    # Saving the calculated angular velocity with corresponding file name
    np.savetxt(folder+'data/analysis.csv', data_write, delimiter =',',fmt="%s")
    return
    
def main():
    """
    Main function for calling the angular_vel_calc() function
    Input is the string for Path to the directory containing videos

    Returns
    -------
    None.

    """
    # String for directory Path
    folder_name = sys.argv[1]
    # Function that calculates and saves angular velocity
    angular_vel_calc(folder_name=folder_name)
    return
    
if __name__ == "__main__":
    main()
