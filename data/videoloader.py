import os.path
import cv2


class CustomLoader:
    def __init__(self, cropxxyy):
        self.cropxx, self.cropyy = cropxxyy

    def load_video(self, path, frame_interval=1):
        assert os.path.exists(path), '%s does not exist.'
        vformat = os.path.splitext(path)[1]
        assert vformat in ['.mp4', '.avi'], 'Unrecognized video format %s' % vformat

        vcap = cv2.VideoCapture(path)
        assert vcap.isOpened(), 'Cannot capture source'

        nfrm = 0
        while vcap.isOpened():
            ret, frame = vcap.read()
            if ret:
                if nfrm % frame_interval == 0:
                    hraw = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    wraw = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    frame = frame[int(hraw * self.cropyy[0]):int(hraw * self.cropyy[1]),
                            int(wraw * self.cropxx[0]):int(wraw * self.cropxx[1]),
                            :]
                    yield frame
                nfrm += 1
            else:
                print('cv2.read() failed')
        vcap.release()
        return None

    def get_frameprops(self, path):
        assert os.path.exists(path), '%s does not exist.' % path
        vformat = os.path.splitext(path)[1]
        assert vformat in ['.mp4', '.avi'], 'Unrecognized video format %s' % vformat

        vcap = cv2.VideoCapture(path)
        assert vcap.isOpened(), 'Cannot capture source'

        vcap.set(cv2.CAP_PROP_CONVERT_RGB, 1)

        return (int(vcap.get(cv2.CAP_PROP_FRAME_COUNT)),
                int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH) * (self.cropxx[1] - self.cropxx[0])),
                int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT) * (self.cropyy[1] - self.cropyy[0])))
