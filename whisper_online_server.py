#!/usr/bin/env python3
from whisper_online import *

import sys
import argparse
import os
import threading

import numpy as np

parser = argparse.ArgumentParser()

# server options
parser.add_argument("--host", type=str, default='localhost')
parser.add_argument("--port", type=int, default=43007)


# options from whisper_online
add_shared_args(parser)
args = parser.parse_args()


# setting whisper object by args 

SAMPLING_RATE = 16000

size = args.model
language = args.lan

t = time.time()
print(f"Loading Whisper {size} model for {language}...",file=sys.stderr,end=" ",flush=True)

if args.backend == "faster-whisper":
    from faster_whisper import WhisperModel
    asr_cls = FasterWhisperASR
else:
    import whisper
    import whisper_timestamped
#    from whisper_timestamped_model import WhisperTimestampedASR
    asr_cls = WhisperTimestampedASR

asr = asr_cls(modelsize=size, lan=language, cache_dir=args.model_cache_dir, model_dir=args.model_dir)

if args.task == "translate":
#    asr.set_translate_task()
    tgt_language = "no"
else:
    tgt_language = "no" #language

e = time.time()
print(f"done. It took {round(e-t,2)} seconds.",file=sys.stderr)

if args.vad:
    print("setting VAD filter",file=sys.stderr)
    asr.use_vad()


min_chunk = args.min_chunk_size

if args.buffer_trimming == "sentence":
    tokenizer = create_tokenizer(tgt_language)
else:
    tokenizer = None



demo_audio_path = "cs-maji-2.16k.wav"
if os.path.exists(demo_audio_path):
    # load the audio into the LRU cache before we start the timer
    a = load_audio_chunk(demo_audio_path,0,1)

    # TODO: it should be tested whether it's meaningful
    # warm up the ASR, because the very first transcribe takes much more time than the other
    asr.transcribe(a)
else:
    print("Whisper is not warmed up",file=sys.stderr)




######### Server objects

import line_packet
import socket

import logging


class Connection:
    '''it wraps conn object'''
    PACKET_SIZE = 65536

    def __init__(self, conn):
        self.conn = conn
        self.last_line = ""
        self.metadata = None

        self.conn.setblocking(True)
        self.read_metadata()


    def send(self, line):
        '''it doesn't send the same line twice, because it was problematic in online-text-flow-events'''
        if line == self.last_line:
            return
        line_packet.send_one_line(self.conn, line)
        self.last_line = line

    def receive_lines(self):
        in_line = line_packet.receive_lines(self.conn)
        return in_line

    def read_metadata(self):
        '''Read the first line of the connection as metadata'''
        metadata_line = self.conn.recv(self.PACKET_SIZE).decode().splitlines()[0]
        self.metadata = metadata_line.strip()  # Assuming metadata is a single line
        logging.info(f'Metadata received: {self.metadata}')

    def non_blocking_receive_audio(self):
        r = self.conn.recv(self.PACKET_SIZE)
        return r


import io
import soundfile

# wraps socket and ASR object, and serves one client connection. 
# next client should be served by a new instance of this object
class ServerProcessor:

    def __init__(self, c, online_asr_proc, min_chunk):
        self.connection = c
        self.online_asr_proc = online_asr_proc
        self.min_chunk = min_chunk
        self.many_nones = 0

        self.last_end = None

    def receive_audio_chunk(self):
        # receive all audio that is available by this time
        # blocks operation if less than self.min_chunk seconds is available
        # unblocks if connection is closed or a chunk is available
        out = []
        while sum(len(x) for x in out) < self.min_chunk*SAMPLING_RATE:
            raw_bytes = self.connection.non_blocking_receive_audio()
            print(raw_bytes[:10])
            print(len(raw_bytes))
            if not raw_bytes:
                break
            sf = soundfile.SoundFile(io.BytesIO(raw_bytes), channels=1,endian="LITTLE",samplerate=SAMPLING_RATE, subtype="PCM_16",format="RAW")
            audio, _ = librosa.load(sf,sr=SAMPLING_RATE, dtype=np.float32)
            print("Length of out buffer is now", len(out))
            out.append(audio)
        if not out:
            return None
        return np.concatenate(out)

    def format_output_transcript(self,o):
        # output format in stdout is like:
        # 0 1720 Takhle to je
        # - the first two words are:
        #    - beg and end timestamp of the text segment, as estimated by Whisper model. The timestamps are not accurate, but they're useful anyway
        # - the next words: segment transcript

        # This function differs from whisper_online.output_transcript in the following:
        # succeeding [beg,end] intervals are not overlapping because ELITR protocol (implemented in online-text-flow events) requires it.
        # Therefore, beg, is max of previous end and current beg outputed by Whisper.
        # Usually it differs negligibly, by appx 20 ms.

        if o[0] is not None:
            beg, end = o[0]*1000,o[1]*1000
            if self.last_end is not None:
                beg = max(beg, self.last_end)

            self.last_end = end
            print("%1.0f %1.0f %s" % (beg,end,o[2]),flush=True,file=sys.stderr)
            return "%1.0f %1.0f %s" % (beg,end,o[2])
        else:
            print(o,file=sys.stderr,flush=True)
            return None

    def send_result(self, o):
        msg = self.format_output_transcript(o)
        if msg is not None:
            self.connection.send(msg)

    def process(self):
        # handle one client connection
        self.online_asr_proc.init()
        while True:
            a = self.receive_audio_chunk()
            if a is None:
                logging.info("break here")
                self.many_nones += 1
                if self.many_nones > 50:
                    break
                continue
            self.online_asr_proc.insert_audio_chunk(a)
            o = self.online_asr_proc.process_iter(self.connection.metadata)
            try:
                self.send_result(o)
            except BrokenPipeError:
                logging.info("broken pipe -- connection closed?")
                break

#        o = online.finish()  # this should be working
#        self.send_result(o)




# Start logging.
level = logging.INFO
logging.basicConfig(level=level, format='whisper-server-%(levelname)s: %(message)s')

# server loop

class ClientThread(threading.Thread):
    def __init__(self, conn, addr, asr, min_chunk, args, tokenizer):
        super().__init__()
        self.conn = conn
        self.addr = addr
        self.online = OnlineASRProcessor(asr,tokenizer,buffer_trimming=(args.buffer_trimming, args.buffer_trimming_sec))
        self.min_chunk = min_chunk

    def run(self):
        logging.info('INFO: Connected to client on {}'.format(self.addr))
        connection = Connection(self.conn)
        logging.info(f'Received ID: {connection.metadata}')
        proc = ServerProcessor(connection, self.online, self.min_chunk)
        proc.process()
        logging.info("CLOSING CONNECTION!")
        self.conn.close()
        logging.info('INFO: Connection to client closed')

def start_server(host, port, asr, min_chunk, args, tokenizer):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen(5) # Listen for multiple connections, not just 1
        logging.info(f'INFO: Listening on {(host, port)}')
        
        try:
            while True:
                conn, addr = s.accept()
                new_thread = ClientThread(conn, addr, asr, min_chunk, args, tokenizer)
                new_thread.start()
        except KeyboardInterrupt:
            logging.info('INFO: Server is shutting down')
        finally:
            logging.info('INFO: Connection closed, terminating.')
            s.close()

# Replace 'args.host', 'args.port', 'online', and 'min_chunk' with the actual values or ways to obtain them
start_server(args.host, args.port, asr, min_chunk, args, tokenizer)
