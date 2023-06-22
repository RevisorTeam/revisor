
![The Revisor logo](docs/images/logo.png)

The Revisor is a computer vision tool that makes elections results transparent and reliable. 
It automatically counts voter turnout of polling stations allowing civil society 
to verify official results.

Read our **[Medium post]** to learn more about the Revisor's architecture.

[![How does the Revisor work?](docs/images/preview_youtube.png)](https://www.youtube.com/watch?v=4KiZuIlc5wk)

Repository contains following modes of the Revisor's core:

| Mode name               | Purpose                           | How to enable _(edit cfg.py)_           |
|-------------------------|-----------------------------------|-----------------------------------------|
| Local                   | Demo and tests                    | Already enabled                         |
| Dataset maker mode      | Gather unlabelled actions dataset | Set `dataset_maker_mode` to `True`      |
| ReID dataset maker mode | Gather unlabelled ReID dataset    | Set `reid_dataset_maker_mode` to `True` |

Mode for processing huge volumes of data is not presented here («api» mode).

### Use case

---
The Revisor was used to process 2018, 2020 and 2021 elections in Russia.

For instance, our team obtained from [Meduza] 17395 camera videos from 9252 polling stations 
totalling more than 1 190 000 hours of the 2021 elections.

On acquired data «API» pipeline of the Revisor was launched. 
Verified counted data of the system showed ~2 million voting actions, 
meanwhile the official voter turnout was ~3.2 million. It means that we have discovered 
~1 million stuffed ballots.

Based on the insights received from these researches [Meduza's material] was written 
about falsifications during Russian elections.


### How to launch?

---
**Preparation**

1. Make sure your [hardware satisfies requirements] and you have all [required source data]. 

2. Install dependencies by following [installation guide].

3. Install process manager to run the Revisor without stopping: supervisor, pm2, etc. _(optional)_

4. Make sure `target_dirs.csv` rows match with camera folder names located in cfg.source_videos_dir directory  _(in case if you have custom input videos)_

5. Edit `cfg.py` _(in case if you are going to launch dataset maker mode or to enable additional visualization features)_

    * Set `dataset_maker_mode` or `reid_dataset_maker_mode` to `True`
    * Set `save_labelled_videos` to `True` if you would like to save videos with visualized tracking data
    * Change `dataset_dir` to your output dataset directory


**Launching**
1. Go the Revisor's directory in terminal.
2. **To process single row** of `target_dirs.csv` _(all videos located inside that folder)_, run:
    
    `python3 revisor.py`
    
    If you launch next iteration, recently processed target directories will be skipped 
    (stored in `cfg.processed_videos_csv`).

    If the Revisor will be stopped during processing, next iteration 
    will be started from the last unfinished video (stored in `cfg.processed_videos_dir`)

3. **To process all rows** of `target_dirs.csv`, run `revisor.py` inside process manager. 
For example, pm2 launching command:

    `pm2 start revisor.py --interpreter python3`

**Outputs**

If you want to see the detections and tracking data, enable drawing of cv2 window:
set `cfg.show_recognitions` to `True`.

All processing results you can find in `data/revisor_results` directory:

* local datetime of each vote action (`stats/votes.csv`)
* processed videos statistics (`stats/stats.csv`)
* samples of the recognized actions (`json` and `videos` directories) - specified in `The Revisor results` section in cfg.py   


### Improvements 

---

We are open to any help. If you want to contribute to the project, we see following things to improve:

- [ ] Optimize YOLACT and QueryInst with TensorRT (fp16 quantization) to speed up ballot boxes recognition module
- [ ] Move all models to the Triton service
- [ ] Call predict from Triton service via gRPC
- [ ] Utilize Docker for dependencies installation and launching the Revisor
- [ ] Experiment with action recognition models in order to increase accuracy 


### Our plans 

---

- [ ] Reveal the Revisor action dataset (ReSet Actions)
- [x] Reveal backbone evolution library (Evolly)
- [ ] Reveal re-identification model optimized by Evolly and training pipelines


### Contacts

---

Contact us if you are interested in cooperating with us, ready to invest 
in us or would like to process your elections with the Revisor: 

**revisorteam@pm.me**


### References 

---

Thanks to following project's authors for their amazing work:
* [EvoPose2D](https://github.com/wmcnally/evopose2d)
* [TensorRT](https://github.com/NVIDIA/TensorRT)
* [TensorRT YOLOv4](https://github.com/jkjung-avt/tensorrt_demos)
* [YOLACT](https://github.com/dbolya/yolact)
* [QueryInst](https://github.com/hustvl/QueryInst)
* [SORT tracker](https://github.com/abewley/sort)
* [Centroid tracker](https://pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/)


[Medium post]: https://medium.com/@RevisorTeam/how-ai-exposed-millions-of-forged-votes-in-russian-elections-40f9bd3fa655
[Meduza]: https://meduza.io/en/
[Meduza's material]: https://meduza.io/en/feature/2022/08/12/17-1-million-stuffed-ballots
[hardware satisfies requirements]: docs/HARDWARE_REQUIREMENTS.MD
[required source data]: docs/DATA_REQUIREMENTS.MD
[installation guide]: docs/INSTALLATION.MD
