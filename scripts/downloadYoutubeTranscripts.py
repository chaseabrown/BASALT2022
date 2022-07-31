from minedojo.data import YouTubeDataset
from youtube_transcript_api import YouTubeTranscriptApi
from tqdm import tqdm
import pandas as pd
from multiprocessing import Process

#Process to be run in parallel
def workerTask(samples, i):
    for video in tqdm(samples):
        try:
            srt = YouTubeTranscriptApi.get_transcript(video["id"])
            df = pd.DataFrame.from_dict(srt)
            df.to_csv("/Volumes/Extreme SSD/Extra Datasets/MineDojo Youtube/transcripts/" + video["id"] + ".csv")
        except:
            pass
            
#Runs body in i parallel fragements
def runBlocks(cores, samples):
    listOfProcesses = []
    for i in range(0, cores):
        start = int(i*(len(samples)/cores))
        end = int((i+1)*(len(samples)/cores))
        process = Process(target=workerTask, args=(samples[start:end], i))
        process.start()
        listOfProcesses.append(process)
    
    for process in listOfProcesses:
        process.join()


def main():
    youtube_dataset = YouTubeDataset(
        full=False,     # full=False for tutorial videos or 
                       # full=True for general gameplay videos
        download=False, # download=True to automatically download data or 
                       # download=False to load data from download_dir
        download_dir="/Volumes/Extreme SSD/Extra Datasets/MineDojo Youtube/" 
                       # default: "~/.minedojo". You can also manually download data from
                       # https://doi.org/10.5281/zenodo.6641142 and put it in download_dir.           
    )
    
    cores = 12
    runBlocks(cores, youtube_dataset)
    
if __name__ == '__main__':
    main()
    