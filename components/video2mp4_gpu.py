import os
import subprocess
import threading
from subprocess import DEVNULL


def gpu_convert_to_mp4(src_video_path, converted_video_path, width=640, height=480, fps=8):
	resolution_str = '{}x{}'.format(width, height)
	fps_str = 'fps={}'.format(fps)
	subprocess.run([
		'ffmpeg', '-y', '-hwaccel', 'cuda', '-resize', resolution_str, '-hwaccel_output_format', 'cuda', '-i',
		src_video_path, '-c:v', 'h264_nvenc', '-b:v', '5M', '-preset', 'fast', '-filter:v', fps_str,
		'-loglevel', 'panic', converted_video_path, '-nostdin'
	])


def call_cuda_ffmpeg(source_filename, start, end, output_filename, concatenate_txt_path, resolution_str, fps_str):

	cmd = [
		'ffmpeg', '-y', '-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda',
		'-i', source_filename, "-ss", start, "-to", end,
		'-vf', 'hwdownload,format=nv12,{}'.format(fps_str),
		'-c:v', 'h264_nvenc', '-b:v', '0', '-vsync', '0',
		'-preset', 'fast', output_filename, '-nostdin', "-loglevel", "docs"
	]

	subprocess.call(cmd, stderr=DEVNULL, stdout=DEVNULL, timeout=120)

	line = "file '{}'\n".format(output_filename)
	with open(concatenate_txt_path, "a") as myfile:
		myfile.write(line)


def sort_concatenated_paths(concatenate_txt_path, sorted_txt_path='tmp_sorted_concatenate.txt'):
	infile = open(concatenate_txt_path)
	lines = []
	for line in infile:
		lines.append(line)
	infile.close()
	lines.sort()
	outfile = open(sorted_txt_path, "w")
	for i in lines:
		outfile.writelines(i)
	outfile.close()
	return sorted_txt_path


def parallel_gpu_conversion(
		src_video_path, converted_video_path, cam_path,
		resolution_str, fps_str, cut_times, concatenate_txt_path='tmp_concatenate.txt'):

	threads = []
	if len(cut_times) == 1:
		time = cut_times[0]
		cmd = [
			'ffmpeg', '-y', '-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda',
			'-i', src_video_path, "-ss", time[0], "-to", time[1],
			'-vf', 'hwdownload,format=nv12,{}'.format(fps_str),
			'-c:v', 'h264_nvenc', '-b:v', '0', '-vsync', '0',
			'-preset', 'fast', converted_video_path, '-nostdin', "-loglevel", "docs"
		]
		subprocess.call(cmd, stderr=DEVNULL, stdout=DEVNULL, timeout=120)

	else:
		open(concatenate_txt_path, 'w').close()
		for idx, time in enumerate(cut_times):
			output_filename = os.path.join(cam_path, 'temp_output{}.mp4'.format(idx))

			thr = threading.Thread(
				target=call_cuda_ffmpeg,
				args=(src_video_path, time[0], time[1], output_filename, concatenate_txt_path, resolution_str, fps_str)
			)
			thr.start()
			threads.append(thr)

		for thr in threads:
			thr.join()

		# Sort .txt files to be in a right order
		sorted_txt_path = sort_concatenated_paths(concatenate_txt_path)

		cmd = [
			"ffmpeg", '-y', "-f", "concat", "-safe", "0", "-i", sorted_txt_path,
			"-c", "copy", converted_video_path, "-loglevel", "quiet"
		]

		subprocess.call(cmd, stderr=DEVNULL, stdout=DEVNULL, timeout=60)

	# Delete temp files
	for file in os.listdir(cam_path):
		if ("temp" in file) and not os.path.isdir(cam_path + file):
			os.remove(os.path.join(cam_path, file))


def get_video_length(filename):
	result = subprocess.run([
		"ffprobe", "-v", "error", "-show_entries", "format=duration",
		"-of", "default=noprint_wrappers=1:nokey=1", filename
	], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	return float(result.stdout)


def get_video_resolution(filename):
	result = subprocess.run([
		"ffprobe", "-v", "error", "-select_streams", "v:0",
		"-show_entries", "stream=width,height", "-of", "csv=p=0",
		filename
	], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	resolution = result.stdout.decode("utf-8").split('\n')[0]
	width = int(resolution.split(',')[0])
	height = int(resolution.split(',')[1])
	return width, height


def get_cut_time(current_cut_secs, part_length_secs):
	current_cut_secs += part_length_secs
	hours = int(current_cut_secs / 3600)
	hours_diff = current_cut_secs - hours * 3600
	minutes = int(hours_diff / 60)
	minutes_diff = hours_diff - minutes * 60
	seconds = int(minutes_diff)
	return current_cut_secs, hours, minutes, seconds


def convert_to_mp4_parallel_gpu(src_video_path, converted_video_path, cam_path, parts=6, width=640, height=480, fps=8):
	resolution_str = '{}x{}'.format(width, height)
	fps_str = 'fps={}'.format(fps)

	video_length_secs = get_video_length(src_video_path)
	part_length_secs = int(video_length_secs) / parts

	current_cut_secs = 0
	cut_times = []
	start_cut_time = "00:00:00"
	for i in range(0, parts):
		current_cut_secs, cut_hours, cut_minutes, cut_seconds = get_cut_time(current_cut_secs, part_length_secs)
		end_cut_time = "{:02d}:{:02d}:{:02d}".format(cut_hours, cut_minutes, cut_seconds)
		cut_times.append([start_cut_time, end_cut_time])
		start_cut_time = end_cut_time

	parallel_gpu_conversion(src_video_path, converted_video_path, cam_path, resolution_str, fps_str, cut_times)
