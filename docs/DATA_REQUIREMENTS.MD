# **The Revisor's** input data requirements

To test our system with demo data you need to [download videos archive](https://drive.proton.me/urls/WQ3SPRX26W#W9VIFkNu3c2b) 
and unpack it to `cfg.source_videos_dir` directory (by default: `data/source_videos`). 
Demo json (`data/stations_demo.json`) is already inside repository.

To run the Revisor with your own data you should create json containing 
polling stations info, rename camera directories and source videos properly:

#### Source videos format

---

**Directories names format**

Each directory located inside cfg.source_videos_dir is a polling station's camera 
folder containing all recorded videos of the corresponding camera. 

Camera directories must be named according to the template:

r{R}u{S}c{N}_{CID}, where:

* {R} - two-digit number of the polling station's region (if exists, otherwise - 00)
* {S} - four-digit number of the polling station
* {N} - one-digit polling station's camera number
* {CID} - camera ID string

For example: 

r78u2046c1_09157441-712a-aaaa-aaaa-64db8b2b414b-sub

**Videos filenames format**

Input videos located inside camera folders must be named according to the template:

{StartingHour}\_{StartingMinute}-{EndingHour}\_{EndingMinute}\_{StartingUnixEpoch}-{EndingUnixEpoch}.mp4

Hours and minutes must be in local time (including leading zeros).

For example:

12_00-12_15_1521363600-1521364500.mp4

**Videos requirements**

Input videos must:
 
* have constant frame rate (fps)
* be in .mp4 format (h264 codec recommended). Audio track is not needed
* have duration ~15-20 minutes long
* be resized to ~307200 pixels in total. Don't forget to resize videos with respect 
to the aspect ratio. _Demo videos are already resized to 640x480 
satisfying requirement: 640*480 = 307200._ 

You can convert and resize your videos to .mp4 utilizing function **convert_to_mp4_parallel_gpu** 
from components/video2mp4_gpu.py which required ffmpeg to be compiled with GPU support 
(for compiling see Step 4 [here](INSTALLATION.MD)).



#### Polling stations info json

---

Path to the polling stations info json you can find in cfg.stations_path.
Example of stations json located in `data/stations_demo.json`.

Content of the demo json:

| Key name                                          | Required* | Data type       |
|---------------------------------------------------|-----------|-----------------|
| **latitude**                                      | -         | float           |
| **longitude**                                     | -         | float           |
| **region**                                        | -         | string          |
| **region_number**                                 | +         | int             |
| **station_address**                               | -         | string          |
| **station_id**                                    | -         | string          |
| **station_number**                                | +         | int             |
| **timezone_offset_minutes**                       | +         | int             |
| **ballots_given_at_station_on_election_day**      | +         | int             |
| **ballots_given_before_election_day**             | +         | int             |
| **ballots_given_outside_station_on_election_day** | +         | int             |
| **turnout**                                       | +         | float, 0...1    |
| **koib_keg**                                      | +         | int, 0 / 1 / 2  |
| **voters_registered**                             | +         | int             |
| **utc8am**                                        | -         | int, unix epoch |
| **camera_id**                                     | +         | list of strings |
| **ballots_in_ballot_boxes_after_election**        | +         | int             |
| **turnout_voters**                                | +         | int             |

\* - Required in local mode

* timezone_offset_minutes - local time offset in minutes from the UTC time
* turnout - voter turnout of polling station (varies from 0 to 1). Formula:  

(ballots_in_ballot_boxes_after_election + ballots_given_outside_station_on_election_day) / voters_registered

* koib_keg - ballot box type. There are 3 classes of ballot boxes in Russia: 
    * 0 - regular ballot box
    * 1 - koib (electronic ballot box)
    * 2 - keg (compact electronic ballot box)

* utc8am - unix epoch of the polling station's 8:00 local time (that is the opening time of the polling station in Russia)
* camera_id - list of strings containing camera IDs (each polling station may have multiple cameras)
* turnout_voters - voter turnout of polling station. Formula: 

ballots_in_ballot_boxes_after_election + ballots_given_outside_station_on_election_day

**Note:**

Stations records in json must match with recorded cameras inside cfg.source_videos_dir folder:

* "region_number" must match with {R}
* "station_number" must match with {S}
* One of the "camera_id" list's element must match with {CID}
