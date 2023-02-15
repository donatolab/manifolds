
###############################################
################ HELPER FUNCTIONS #############
###############################################

def get_mouse_ids(header,
                  dfs):
    
    #
    header_ = np.array(header)

    idx = np.where(header_=='mouse ID')[0][0]

    ids = np.vstack(df.iloc[:,idx])

    return ids.squeeze()

def get_file_paths(header,
                  dfs):
    
    #
    header_ = np.array(header)

    idx = np.where(header_=='path')[0][0]

    paths = np.vstack(df.iloc[:,idx])

    return paths.squeeze()

def get_sessions(header,
                 dfs):
    
    #
    header_ = np.array(header)

    idx = np.where(header_=='sessions')[0][0]

    sessions = np.vstack(df.iloc[:,idx][:])

    return sessions.squeeze()

def convert_mesc_sessions_to_concatenated_tiff(fname, 
                                               sessions_in):
    
    fname_out = fname.replace('.mesc','.tif')

    if os.path.exists(fname_out):
        return fname_out
    
    sessions_in = sessions_in.replace(" ", "").split (",")

    #
    sess_list = [] 
    for session in sessions_in:
        temp = session.replace("S",'')
        temp = 'MUnit_'+str(int(temp)-1)
        print ("session loaded: ", temp)
        sess_list.append(temp)
    #
    data = []
    with h5py.File(fname, 'r') as file:
                #
        for sess in sess_list:
            print ("processing: ", sess)
            temp = file['MSession_0'][sess]['Channel_0'][()]
            print ("    data loaded size: ", temp.shape)
            data.append(temp)

    data = np.vstack(data)
    print(data.shape)

    # from tifffile import tifffile
    
    tifffile.imwrite(fname_out, data)
    
    return fname_out


def run_suite2p_from_fname(fname):

    data = imread(fname)
    print('imaging data of shape: ', data.shape)
    n_time, Ly, Lx = data.shape

    #
    ops = suite2p.default_ops()
    ops['batch_size'] = 200 # we will decrease the batch_size in case low RAM on computer
    ops['threshold_scaling'] = 2.0 # we are increasing the threshold for finding ROIs to limit the number of non-cell ROIs found (sometimes useful in gcamp injections)
    ops['fs'] = 30 # sampling rate of recording, determines binning for cell detection
    ops['tau'] = 1.25 # timescale of gcamp to use for deconvolution
    ops['save_path0'] = os.path.split(fname)[0]
    ops['save_folder'] = os.path.split(fname)[0]
    ops['tiff_list'] = [fname]

    print(ops)

    #
    db = {
        'data_path': os.path.split(fname)[0],
    }
    print(db)

    #
    output_ops = suite2p.run_s2p(ops=ops, db=db)

      