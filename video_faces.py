import onnxruntime
import cv2
from mtcnn_ort import MTCNN
import numpy as np
from skimage import transform as trans
import argparse
import onnxruntime as ort
import numpy as np
from scipy.spatial import distance


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True, help='Video file file to process.', )
    parser.add_argument('--model', required=True, help='File with text corpora.', )
    parser.add_argument('--resolution', default=64, type=int, help='Resolution of facial crops.')
    parser.add_argument('--similarity-threshold', default=0.5, type=float)
    parser.add_argument('--detection-threshold', default=0.95, type=float)
    parser.add_argument('--input-scale', default=0.5, type=float)

    args = parser.parse_args()
    return args


class FaceDB:
    def __init__(self, model, resolution, similarity_threshold):
        self.model = model
        self.resolution = resolution
        self.similarity_threshold = similarity_threshold

        self.template = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        self.template /= 112
        self.template = (self.template - 0.5) * 1.1 + 0.5
        self.template[:, 1] -= 0.1
        self.template *= self.resolution
        self.tform = trans.SimilarityTransform()

        self.ort_sess = ort.InferenceSession(self.model,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        test_data = np.zeros((1, 3, self.resolution, self.resolution), dtype=np.float32)
        outputs = self.ort_sess.run(None, {'input_image': test_data})[0]
        self.emb_dim = outputs.shape[1]
        self.max_faces = 1000
        self.faces = []
        self.store_resolution = 32
        self.face_emb = np.ones((self.max_faces, self.emb_dim), dtype=np.float32)

    def add_face(self, frame, landmarks, threshold=10):
        self.tform.estimate(landmarks, self.template)
        M = self.tform.params[0:2, :]
        crop = cv2.warpAffine(frame, M, (self.resolution, self.resolution), borderValue=0.0)
        #crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_data = np.transpose(crop, (2,0,1))[np.newaxis].astype(np.float32) / 255.0
        embedding = self.ort_sess.run(None, {'input_image': crop_data})
        print(embedding[0].dot(embedding[0].T))
        embedding = embedding[0]

        crop = cv2.resize(crop, (self.store_resolution, self.store_resolution))
        if len(self.faces) > 0:
            sim = embedding.dot(self.face_emb[:len(self.faces)].T)
            sim = sim[0]
            #distances = distance.cdist(embedding, self.face_emb[:len(self.faces)], metric='cosine')
            #distances = distances[0]
            best_id = np.argmax(sim)
            print(sim[best_id], best_id)
            if sim[best_id] > self.similarity_threshold:
                self.faces[best_id].append(crop)
            elif sim[best_id] < self.similarity_threshold * 0.9:
                self.face_emb[len(self.faces)] = embedding
                self.faces.append([crop])
        else:
            self.face_emb[len(self.faces)] = embedding
            self.faces.append([crop])

    def draw_faces(self, max_faces=20, max_rows=32):
        if not self.faces:
            return None
        lines = [face[::-1][:max_faces] for face in self.faces]
        lines = [line + [np.zeros((self.store_resolution, self.store_resolution, 3), dtype=np.uint8)] * max(0, max_faces - len(line)) for line in lines]
        lines = [np.concatenate(line, axis=1) for line in lines]
        columns = []
        while lines:
            columns.append(lines[:max_rows])
            lines = lines[max_rows:]

        for column in columns[1:]:
            if len(column) < len(columns[0]):
                column += [column[-1]] * (len(columns[0]) - len(column))
        columns = [np.concatenate(column, axis=0) for column in columns]
        return np.concatenate(columns, axis=1)


def main():
    args = parse_arguments()
    detector = MTCNN()
    face_database = FaceDB(args.model, args.resolution, args.similarity_threshold)
    capture = cv2.VideoCapture(args.video)

    while capture.isOpened():
        for i in range(10):
            ret, frame = capture.read()
            if not ret:
                break

        if args.input_scale != 1.0:
            frame = cv2.resize(frame, (0,0), fx=args.input_scale, fy=args.input_scale, interpolation=cv2.INTER_AREA)

        all_bb, all_landmarks = detector.detect_faces_raw(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        all_landmarks = all_landmarks.reshape(2, 5, -1)
        all_landmarks = np.transpose(all_landmarks, (2, 1, 0))
        all_landmarks = [l for l, b in zip(all_landmarks, all_bb) if b[4] > args.detection_threshold]

        for landmarks in all_landmarks:
            face_database.add_face(frame, landmarks)

        collage = face_database.draw_faces()
        if collage is not None:
            cv2.imshow('faces', collage)

        cv2.imshow('video', frame)
        key = cv2.waitKey(10)
        if key == 27:
            break

        #detector.detect_faces(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


if __name__ == "__main__":
    main()