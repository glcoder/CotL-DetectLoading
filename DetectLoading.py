import os
import ffmpeg
import numpy as np
import cv2 as cv
from concurrent.futures import ThreadPoolExecutor
from argparse import ArgumentParser

def transform(image):
    return image
    #return cv.Canny(image, 50, 200)
    #return cv.bitwise_not(cv.Canny(image, 50, 200))

def detect(frame, scaled):
    image = transform(frame['image'])
    for template in scaled:
        result = cv.matchTemplate(image, template['image'], cv.TM_CCOEFF_NORMED)
        _, threshold, _, position = cv.minMaxLoc(result)
        if threshold >= 0.75:
            margin = int(template['scale'] * 5)
            return {
                'image': template['image'],
                'width': template['width'],
                'height': template['height'],
                'localtion': (
                    position[0] - margin,
                    position[1] - margin,
                    position[0] + margin + template['width'],
                    position[1] + margin + template['height'],
                )
            }
    return None

def matching(save, nframe, frame, detected, loading):
    x1, y1, x2, y2 = detected['localtion']
    image = transform(frame['image'][y1:y2,x1:x2])
    result = cv.matchTemplate(image, detected['image'], cv.TM_CCOEFF_NORMED)
    _, threshold, _, _ = cv.minMaxLoc(result)
    if threshold >= 0.75:
        loading.append(nframe)
        if save: cv.imwrite(f'frames/{nframe:07}_{threshold:.2f}.jpg', image)

def process_video(save, video, template):
    scaled = []
    detected = None
    nframe = 0
    loading = []

    for scale in np.linspace(1.0, 0.5, 20):
        size = (int(template['width'] * scale), int(template['height'] * scale))
        image = cv.resize(template['image'], size, interpolation = cv.INTER_CUBIC)
        scaled.append({
            'image': image,
            'scale': scale,
            'width': image.shape[1],
            'height': image.shape[0],
        })

    process = (
        ffmpeg
            .input(video['name'])
            .output('pipe:', format='rawvideo', pix_fmt='gray')
            .run_async(pipe_stdout=True)
    )

    with ThreadPoolExecutor() as executor:
        while True:
            data = process.stdout.read(video['width'] * video['height'] * 1)
            if not data:
                break

            nframe = nframe + 1
            if nframe < video['start']:
                continue

            image = np.frombuffer(data, np.ubyte).reshape(video['height'], video['width'], 1)
            frame = {
                'image': image,
                'width': image.shape[1],
                'height': image.shape[0],
            }

            if detected is None:
                detected = detect(frame, scaled)

            if detected is not None:
                executor.submit(matching, save, nframe, frame, detected, loading)

    process.stdout.close()
    process.wait()

    return loading

def main(options):
    if options.frames:
        os.makedirs('frames', exist_ok=True)

    image = cv.imread(options.template, cv.IMREAD_GRAYSCALE)
    template = {
        'image': transform(image),
        'width': image.shape[1],
        'height': image.shape[0],
    }

    probe = ffmpeg.probe(options.input, select_streams='v')
    video = {
        'name': options.input,
        'start': int(options.start),
        'width': int(probe['streams'][0]['width']),
        'height': int(probe['streams'][0]['height']),
        'framerate': int(probe['streams'][0]['r_frame_rate'][:-2]),
    }

    loading = sorted(process_video(options.frames, video, template))
    if len(loading) == 0:
        return

    print('start;frames;seconds')

    start = 0
    for i in range(1, len(loading)):
        if loading[i] != loading[i-1] + 1:
            duration = i - start
            seconds = duration / video['framerate']
            print(f'{loading[start]};{duration};{seconds:.3f}')
            start = i

    duration = len(loading) - start
    seconds = duration / video['framerate']
    print(f'{loading[start]};{duration};{seconds:.3f}')

    seconds = len(loading) / video['framerate']
    print(f'0;{len(loading)};{seconds:.3f}')
    

if __name__ == '__main__':
    parser = ArgumentParser(description='Cult of the Lamb speedrun loading detector')
    parser.add_argument('-i', '--input', type=str, required=True, help='input video')
    parser.add_argument('-s', '--start', type=int, default=0, help='speedrun start frams')
    parser.add_argument('-t', '--template', type=str, default='template.png', help='template image')
    parser.add_argument('-f', '--frames', action='store_true', help='save loading frames')
    main(parser.parse_args())
