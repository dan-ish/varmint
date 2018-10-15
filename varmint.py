import container as contain
import controller as control
import sys
import os
import datetime
import time


"""
VARMINT
N quit_time save_time num_threads config_path

Config file should be
err_goal min_runs max_runs max_bad_runs numT Tstep

quit_time = quit time in hours
save_time = save time in minutes

"""



message_path = "/home/dish/varmint/messages/"
run_exception_limit = 100
sleep_time = 0.1



def write_message(message,file_name=None,quit_time=-1,save_time=-1):
    if file_name==None:
        mess_file = open(message_path+"global.m",'a')
    else:
        mess_file = open(message_path+file_name+'.m','a')
    now = datetime.datetime.now()
    mess_file.write("{0}/{1}/{2} at {3:02}:{4:02}:  ".format(now.month,now.day,now.year,now.hour,now.minute))
    mess_file.write(message)
    if save_time>0:
        save_date = datetime.datetime.fromtimestamp(save_time)
        mess_file.write("\nNext save at {0:02}:{1:02} on {2}/{3}/{4}"
                        .format(save_date.hour,save_date.minute,
                                save_date.month,save_date.day,save_date.year))
    if quit_time>0:
        quit_date = datetime.datetime.fromtimestamp(quit_time)
        mess_file.write("\nForce exit at {0:02}:{1:02} on {2}/{3}/{4}"
                        .format(quit_date.hour,quit_date.minute,
                                quit_date.month,quit_date.day,quit_date.year))
    mess_file.write("\n\n\n")
    mess_file.close()

def main():

    try:
        os.mkdir(message_path)
    except OSError:
        pass

    try:
        os.mkdir(contain.data_path)
    except OSError:
        pass

    #Resolving the command line arguments
    if len(sys.argv)<7:
        write_message("Too few arguments, exiting.")
        return

    try:
        N = int(sys.argv[1])
    except Exception as e:
        write_message("Failed to resolve N. Exception:\n"+repr(e))
        return

    if N <=0 or not N%2==0:
        write_message("N must be a positive even integer, exiting")
        return

    try:
        u = float(sys.argv[2])
    except Exception as e:
        write_message("Failed to resolve u. Exception:\n"+repr(e))
        return

    try:
        quit_time = float(sys.argv[3])
    except Exception as e:
        write_message("Failed to resolve maximum execution time. Exception:\n"+repr(e),str(N)+"_"+str(u))
        return
    if quit_time<=0:
        write_message("Time limit for execution must be positive, exiting",str(N)+"_"+str(u))
        return
    
    try:
        save_time = float(sys.argv[4])
    except Exception as e:
        write_message("Failed to resolve time between saves. Exception:\n"+repr(e),str(N)+"_"+str(u))
        return
    if save_time<=0:
        write_message("Time between saves must be positive, exiting",str(N)+"_"+str(u))
        return

    try:
        num_threads = int(sys.argv[5])
    except Exception as e:
        write_message("Failed to resolve number of worker threads. Exception:\n"+repr(e),str(N)+"_"+str(u))
        return

    try:
        config = open(sys.argv[6],'r')
    except Exception as e:
        write_message("Failed to open config file: \""+sys.argv[6]+"\". Exception:\n"+repr(e),str(N)+"_"+str(u))
        return

    options = config.readline().split()
    config.close()
    try:
        runs_req = int(reduce(lambda x,y:x+y,options[0].split(',')))
        max_bad_runs = int(reduce(lambda x,y:x+y,options[1].split(',')))
        numT = int(reduce(lambda x,y:x+y,options[2].split(',')))
        Tstep = float(options[3])
    except Exception as e:
        write_message("Failed to resolve control parameters from config file. Exception:\n"+repr(e),str(N)+"_"+str(u))
        return
    if runs_req<=0:
        write_message("Requested number of runs must be positive, exiting",str(N)+"_"+str(u))
        return
    if max_bad_runs<=0:
        write_message("Maximum number of failed runs must be positive, exiting",str(N)+"_"+str(u))
        return
    if Tstep<=0:
        write_message("Temperature step size must be positive, exiting.",str(N)+"_"+str(u))

    abs_quit_time  = time.time()+3600*quit_time #Absolute quit time
    abs_save_time = min(abs_quit_time,time.time()+60*save_time)#Absolute save time

    try:
        #loading any saves
        if not os.path.isfile(contain.data_path+"{0}_{1}.d".format(N,u)):
            bundle = contain.Container(N,u,Tstep,numT,runs_req,max_bad_runs) 
            bundle.save_self('1')
        else:
            try:
                bundle = contain.open_data(N,u)
            except Exception as e:
                if str(e) == 'open':
                    write_message("An instance of Varmint is already running at N = {0}".format(N),str(N)+"_"+str(u))
                elif str(e) == 'flag':
                    write_message("No valid flag on save file, exiting",str(N)+"_"+str(u))
                else:
                    write_message("Could not load saved data. Exception:\n"+repr(e),str(N)+"_"+str(u))
                return
            
            bundle.set_stop_cond(runs_req,max_bad_runs)

            if not bundle.more_runs(abs_quit_time):
                write_message("No runs remain with current settings, exiting",str(N)+"_"+str(u))
                bundle.save_self('0')
                return

            bundle.save_self('1')

    except Exception as e:
        write_message("Exception encountered while initializing data container:\n"+repr(e),str(N)+"_"+str(u))
        return

    write_message("Successful startup, running with {0} threads".format(num_threads)
                ,str(N)+"_"+str(u),abs_quit_time,abs_save_time)


    try:

        griddle = control.Controller(num_threads,bundle)
        while True:
            time.sleep(sleep_time)
            griddle.flip_runs(abs_quit_time)
            if not griddle.runs_remaining():
                griddle.finish()
                message = "All requested runs complete. Final status:\n"+bundle.status()
                write_message(message,str(N)+"_"+str(u))
                bundle.save_self('0')
                return

            if time.time()>= abs_quit_time:
                griddle.abandon()
                message = "Time limit reached. Final status:\n"+bundle.status()
                write_message(message,str(N)+"_"+str(u))
                bundle.save_self('0')
                return

            if time.time() >= abs_save_time:
                bundle.save_self('1')
                message = "Data saved by timer. Status:\n"+bundle.status()
                abs_save_time = min(abs_quit_time,time.time()+60*save_time)
                write_message(message,str(N)+"_"+str(u),abs_quit_time,abs_save_time)
                    
    except Exception as e:
        message = "Exception encountered while running minimizer:\n"+repr(e)+"\n"
        message = message+"Dumping data to file dump_"+str(N)+" and exiting."
        try:
            bundle.dump_self()
            griddle.abandon()
        except Exception as e:
            messages = message + " Data dump failed:\n"+repr(e)
        write_message(message,str(N)+"_"+str(u))
        return


if __name__ == "__main__":
    main() 