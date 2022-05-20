"""
Import necessary libraries to run this program
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

def vid_load(fname):
    """
    The fucntion for loading video files and saving as np array. This function
    requires that the video format be compatible with cv2
    
    Inputs
    -------
    fname (str) : PATH to the video directory for loading the files
    
    Returns
    -------
    video (np.array): np array of video data with shape 
                      (frames, height, width, channels)

    """
    
    cap = cv2.VideoCapture(fname)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video = np.empty((frameCount, frameHeight, frameWidth, 3), 
                     np.dtype('uint8'))
    fc = 0
    ret = True
    while (fc < frameCount  and ret):
        ret, video[fc,:,:,:] = cap.read()
        fc += 1
    cap.release()
    return video

def hist_eq(frame):
    """
    Function for impleneting CLAHE on individual images
    
    Inputs
    -------
    frame (np.array) : image with shape (height, width, channels=3)
    
    Returns
    -------
    final (np.array) : image with shape (height, width, channels=3)

    """
    
    frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l,a,b=cv2.split(frame_lab)
    clahe = cv2.createCLAHE(clipLimit=20.0,tileGridSize=(5,5))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def extract_organoid_semi(video,pad):
    """
    Function for implementing cv2 selectROI and cropping the video for region
    of interest
    
    Inputs
    -------
    video (np.array): np array of video data with shape 
                      (frames, height, width, channels)
    pad (int): Extra area around selected ROI to be included in the cropped 
               video
    
    Returns
    -------
    video_out (np.array): np array of video data with shape determined by 
                      selected region

    """
    
    frame = video[0,:,:,:]
    r = cv2.selectROI(frame)
    video_out = video[:,int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2]),:]
    return video_out

def get_corresp_grid(video, spacing):
    """
    Function for generating a grid of correspondences on a video
    
    Inputs
    -------
    video (np.array): np array of video data with shape 
                      (frames, height, width, channels)
    spacing (int): the spacing in pixels to be placed between each 
                   correspondence, higher values can decrease sensitivity but 
                   help with speed
    
    Returns
    -------
    out_arr (np.array): an array of size (N,2) where N is the number of 
                        coorespondences

    """
    
    h = video.shape[1]
    w = video.shape[2]
    y = np.arange(0, h, spacing)
    x = np.arange(0, w, spacing)
    xx, yy = np.meshgrid(x, y)
    points = np.array([xx.flatten(),yy.flatten()])
    d, w, h, channel = video.shape
    x = points[0,:]
    y = points[1,:]
    k_x = h//2
    k_y = w//2
    a_2 = (h//2)**2
    b_2 = (w//2)**2
    mask = (((x[:]-k_x)**2)/a_2 + ((y[:]-k_y)**2)/b_2 < 1).T
    bad_pts = np.where(mask==False)[0]
    points_out = np.delete(points,bad_pts,axis = 1)
    out_arr = np.array(points_out.T, dtype = np.float32)
    return out_arr

def mask_ellipse(video_inp):
    """
    Function for creating an ellipse to remove correspondences in corner and
    account for spherical geometry of organoids.
    
    Inputs
    -------
    video_inp (np.array): np array of video data of tracking lines with shape 
                          (frames, height, width, channels)
    
    Returns
    -------
    show_vid (np.array): np array of video data of tracking lines with shape 
                         (frames, height, width, channels)

    """
    
    d, w, h, channel = video_inp.shape
    ellipse = np.zeros((h,w),np.dtype('uint8'))
    x = np.arange(0,h)
    y = np.arange(0,w)
    k_x = h//2
    k_y = w//2
    a_2 = (h//2)**2
    b_2 = (w//2)**2
    mask = (((x[np.newaxis,:]-k_x)**2)/a_2 + 
            ((y[:,np.newaxis]-k_y)**2)/b_2 < 1).T
    ellipse[mask]= 1
    ellipse = ellipse.T
    ellipse_mask = np.repeat(ellipse[:,:,np.newaxis],channel, axis =2)
    ellipse_mask = np.repeat(ellipse_mask[np.newaxis,:,:],d, axis =0)
    video_out = video_inp * ellipse_mask
    outliers = ellipse==0
    outlier_vid = np.repeat(outliers[:,:,np.newaxis],channel, axis = 2)
    outlier_vid = np.repeat(outlier_vid[np.newaxis,:,:],d, axis =0)
    show_vid = video_out
    show_vid[outlier_vid ] = 255
    return show_vid

def video_save(video_inp,iter_num):
    """
    Function for writing the tracking images into the current working 
    directory. Saves files as png images with index as file name
    
    Inputs
    -------
    video_inp (np.array): np array of video data of tracking lines with shape 
                          (frames, height, width, channels)
    iter_num (int): An identifier index of filename to be saved
    
    Returns
    -------
    None.

    """
    
    for i in range(1,video_inp.shape[0]-1):
        cv2.imwrite("video_out/"+str(iter_num)+'_'+str(i)+'.png', 
                    video_inp[i,:,:,:])
    # command for converting saved png images into a video for visualization
#    command = 'ffmpeg -f image2 -r 2 -i %d.png -vcodec mpeg4 -y movie.mp4'
    return 
    
def tracking_LK(video, points, grid_spacing,iter_num,display=False):
    """
    Function for implementing sparse optical flow tacking
    
    Inputs
    -------
    video_inp (np.array): np array of video data of tracking lines with shape 
                          (frames, height, width, channels)
    points (np.array): an array of size (N,2) where N is the number of 
                       coorespondences
    grid_spacing (int): the spacing in pixels to be placed between each 
                        correspondence, higher values can decrease sensitivity 
                        but help with speed
    iter_num (int): An identifier index of filename to be saved
    display (bool): A flag for displaying the tracking lines on video
    
    Returns
    -------
    coordinates (np.array): The pixel coordinates of each correspondence.
                            (N,2,Frequency of computation)

    """
    # parameters for LK tracking function
    lk_params = dict(winSize = (grid_spacing+grid_spacing+10,
                                grid_spacing+grid_spacing+10),
                 maxLevel = 4,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                             20, 0.01))
    # initialization of data arrays
    coordinates = np.zeros((points.shape[0],2,0))
    disp_clean_vid = np.copy(video)
    #filter for abnormaly selected ROI
    if video.shape[1] > 50:
        for num_frames in range(video.shape[0]-1):
            # histogram equalize the frames
            ini_frame = cv2.cvtColor(hist_eq(video[num_frames,:,:,:]), 
                                     cv2.COLOR_BGR2GRAY)
            next_frame = cv2.cvtColor(hist_eq(video[num_frames+1,:,:,:]), 
                                      cv2.COLOR_BGR2GRAY)
            # function for optical flow tracking of correspondences
            new_points, status, error = cv2.calcOpticalFlowPyrLK(ini_frame, 
                                                                 next_frame, 
                                                                 points, 
                                                                 None, 
                                                                 **lk_params)
            # updates to variable for next state and saving new coordinates
            ini_frame = next_frame.copy()
            points = new_points
            x, y = new_points[:,0],new_points[:,1]
            append_pts = np.array([x,y])
            coordinates= np.dstack((coordinates,append_pts.T))
            # Display argument
            if display == True:
                #add tracking line to each correspondence
                for p in range(2,coordinates.shape[2]):
                    for q in range(coordinates.shape[0]):
                        cv2.line(video[num_frames,:,:,:],
                                 (int(coordinates[q,0,p-1]),
                                  int(coordinates[q,1,p-1])),
                                 (int(coordinates[q,0,p]),
                                  int(coordinates[q,1,p])),
                                 (0,255,0),1)
                #create display image
                displayimg = np.concatenate((disp_clean_vid, video),axis =2)
                cv2.imshow("Show",displayimg[num_frames,:,:,:])
            cv2.waitKey(1)
        # Optional: save tracking lines as images for making video
        # video_save(np.uint8(displayimg),iter_num)
    return coordinates

def rot_vel_calc(distance,organoid,display=False):
    """
    Function for calculation of rotational velocity from distance covered by 
    correspondence. Also filters out non moving correpsondences and jumping 
    correspondences.
    
    Inputs
    -------
    distance (np.array) : The pixel coordinates of each correspondence.
                          (N,2,Frequency of computation)
    organoid (np.array): np array of video data of tracking lines with shape 
                         (frames, height, width, channels)
    display (bool): A flag for displaying the histogram of angular velocities
    
    Returns
    -------
    avg_rot_vel (np.array): Array of angular velocities for all 
                            correspondences in the ROI averaged over all 
                            frames, with shape (N,)
    inst_rot_vel (np.array): Array of angular velocities of all 
                             correspondences in the ROI, with shape 
                             (N, frequency of computation)
    
    """
    
    # filter out nan and inf values
    distance = np.nan_to_num(distance, nan = 0.0, posinf= 1 , neginf = 0)
    # determine centroid of organoid for conversion of rotational velocity
    # to angular velocity
    centroid = [organoid.shape[1]/2,organoid.shape[2]/2]
    # calculate distance covered in successive frames
    dist_covered_temp = distance[:,:,0] - distance[:,:,-1]
    dist_covered = np.sqrt(dist_covered_temp[:,0]**2 + 
                           dist_covered_temp[:,1]**2)
    # determine where distance covered is more than the radius of organoid in 
    # one frame
    outlier_upper = np.where(dist_covered > np.max((organoid.shape[1]/2,
                                                    organoid.shape[2]/2)))[0]
    # determine where correspondence is nearly static, < 10 pixels
    outlier_lower = np.where(dist_covered < 10)[0]
    # make list of outlier organoids
    outlier = np.append(outlier_upper, outlier_lower)
    # remove outlier correspondences
    dist_covered_filt = np.delete(dist_covered,outlier)
    distance_filt = np.delete(distance,outlier,axis = 0)
    # check if any correspondences moved or not
    if outlier.shape[0] > dist_covered.shape[0]*0.75:
        # if minimal/no rotation is observed then set output variables = 0
        avg_rot_vel = 0 
        inst_rot_vel = 0
        print("No significant rotation detected")
    else:
        # if rotation is observed
        # calculate the distance from center of organoid
        orig_centre_dist_x = distance_filt[:,0,:]-centroid[0]
        orig_centre_dist_y = distance_filt[:,1,:]-centroid[1]
        orig_centre_dist = np.array([orig_centre_dist_x,orig_centre_dist_y])
        
        # initialize the holder varaible for angular velocity 
        inst_rot_vel = np.empty((orig_centre_dist.shape[1],
                                 orig_centre_dist.shape[2]-1))
        # loop over all correspondences
        for i in range(orig_centre_dist.shape[2]-1):
            x_coord = orig_centre_dist[0,:,i] - orig_centre_dist[0,:,i+1]
            y_coord = orig_centre_dist[1,:,i] - orig_centre_dist[1,:,i+1]
            # calculate velocity in pixel/frame
            vel = np.sqrt(x_coord**2 + y_coord**2)
            # convert rotational velocity to angular velocity by dividing with
            # radial distance
            r_vect = np.sqrt(orig_centre_dist[0,:,i]**2 + 
                             orig_centre_dist[1,:,i]**2)
            inst_rot_vel[:,i] = vel/r_vect
        avg_rot_vel = np.mean(inst_rot_vel,axis = 1)
    # display histogram of rotational and angular velocities
    if display == True:
        plt.hist(dist_covered, bins = 50, alpha = 0.5, label = 'raw')
        plt.hist(dist_covered_filt,bins =50, alpha = 0.8,label = "filtered")
        plt.legend()
        plt.xlabel('distance covered by # of correspondence')
        plt.ylabel('# of correspondence')
        plt.show()
    return avg_rot_vel, inst_rot_vel

def rot_vel_semi(video):
    """
    Function called by main script to calculate angular velocity from videos.
    Also defines the hyper parameters for video tracking
    
    Inputs
    -------
    video_inp (np.array): np array of video data of tracking lines with shape 
                          (frames, height, width, channels)
    
    Returns
    -------
    out_value (float): the angular velocity corresponding to the input video

    """
    
    freq_compute = 25 ### how frequently to compute correspondences
    vid_len = (video.shape[0]//freq_compute)*freq_compute
    pad = -15 ### dilate the bounding box by this constant along H, W
    grid_spacing = 5 ### the space between correspondences to track in the video
    display_tracking = True ### argument to display tracking
    display_histogram = False ### argument to display histogram of velocities
    rot_vel_freq = []
    big_hold = np.empty(0)
    organoid_whole = extract_organoid_semi(video,pad) ### extracts organoid
    organoid = mask_ellipse(organoid_whole)
    iter_num = 0
    for f in range(0,vid_len,freq_compute):
        #generates a correspondence grid for ROI
        corresp = get_corresp_grid(organoid[f:f+freq_compute,:,:],
                                           grid_spacing)
        #calculates the distance covered by each correspondence
        distances = tracking_LK(organoid[f:f+freq_compute,:,:], 
                                corresp, grid_spacing,iter_num,
                                display_tracking)
        #append data to appropriate arrays
        #check if organoid was static or not
        if distances.ndim==3 and distances.shape[0]>1:
            #calculate angular velocity for each correspondence
            rot_vel,inst_vel = rot_vel_calc(distances,
                                            organoid[f:f+freq_compute,:,:],
                                            display_histogram)
            rot_vel_freq.append(np.mean(rot_vel))
            
            if np.array(inst_vel).ndim !=0 :
                big_hold = np.concatenate((big_hold,
                                           np.mean(inst_vel,axis = 0)))
            else:
                big_hold = np.concatenate((big_hold,np.zeros(1))) 
        iter_num +=f
    #Optional: save the data as npy files for visualization of instantaneous 
    #velocity
    # np.save('data/out',big_hold)
    
    # output mean value of angular velcity for entire video
    out_value = np.mean(rot_vel_freq)
    return out_value