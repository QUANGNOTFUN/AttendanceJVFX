package app;

import java.io.File;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

public class Attendance {

    private static final String CASCADE_PATH = "open_cv/haarcascade_frontalface_alt.xml";
    private static final String STUDENT_DATA_PATH = "dataset";
    private CascadeClassifier faceClassifier;
    private Map<Integer, String> studentNames;
    private Map<Integer, Mat> faceHistograms;

    // Adjusting threshold for 80% continuous recognition
    private static final int SUCCESS_THRESHOLD = 160; // 80% match (200 - 80% of 200)
    private static final long DURATION = 3000; // 3 seconds of continuous detection
    private long startTime = 0; // Track time when recognition starts
    private boolean trackingRecognition = false; // Flag to track whether we're counting time
    private long detectionStartTime = 0;

    public Attendance() {
        faceClassifier = new CascadeClassifier(CASCADE_PATH);
        studentNames = new HashMap<>();
        faceHistograms = new HashMap<>();
        loadStudentData();
    }

    public void startRecognition() {
        System.out.println("Bắt đầu nhận diện khuôn mặt...");

        VideoCapture camera = new VideoCapture(0);
        if (!camera.isOpened()) {
            System.out.println("Không thể mở camera.");
            return;
        }

        Mat frame = new Mat();
        while (true) {
            if (camera.read(frame)) {
                Core.flip(frame, frame, 1);
                long currentTime = System.currentTimeMillis();
                if (currentTime - detectionStartTime >= 1000) {
                    detectAndDisplay(frame);
                    detectionStartTime = currentTime;
                }
                HighGui.imshow("Nhận diện khuôn mặt", frame);
                if (HighGui.waitKey(1) == 'q') {
                    break;
                }
            } else {
                System.out.println("Không thể đọc khung hình.");
                break;
            }
        }

        camera.release();
        HighGui.destroyAllWindows();
    }

    private void detectAndDisplay(Mat frame) {
        Mat grayFrame = new Mat();
        Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);
        Imgproc.equalizeHist(grayFrame, grayFrame);

        MatOfRect faces = new MatOfRect();
        faceClassifier.detectMultiScale(grayFrame, faces, 1.2, 5, 0, new Size(100, 100), new Size());

        Rect[] faceArray = faces.toArray();
        for (Rect face : faceArray) {
            Imgproc.rectangle(frame, face.tl(), face.br(), new Scalar(0, 255, 0), 2);

            Map.Entry<Integer, Double> result = recognizeFace(face, frame);
            int studentId = result.getKey();
            double matchValue = result.getValue();

            if (studentId != -1) {
                String studentName = studentNames.get(studentId);
                String accuracyText = String.format("Accuracy: %.0f%%", (100 - matchValue / 200.0 * 100));

                Imgproc.putText(frame, studentName, new Point(face.x, face.y - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0, 255, 0), 2);
                Imgproc.putText(frame, accuracyText, new Point(face.x, face.y + face.height + 20), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0, 255, 0), 2);

                if (matchValue < SUCCESS_THRESHOLD) {  // Accuracy over 80% condition
                    trackRecognition(studentId, matchValue);
                } else {
                    resetRecognition();
                }
            } else {
                Imgproc.putText(frame, "Unknown", new Point(face.x, face.y - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0, 0, 255), 2);
                resetRecognition();
            }
        }
    }

    // Updated method to track continuous recognition
    private void trackRecognition(int studentId, double matchValue) {
        if (!trackingRecognition) {
            trackingRecognition = true;
            startTime = System.currentTimeMillis();
        }

        // Check if 3 seconds of continuous recognition has passed
        if (System.currentTimeMillis() - startTime >= DURATION) {
            updateAttendance(studentId);
            resetRecognition();
        }
    }

    private void resetRecognition() {
        trackingRecognition = false;
        startTime = 0;
    }

    private Map.Entry<Integer, Double> recognizeFace(Rect face, Mat frame) {
        Mat faceMat = new Mat(frame, face);
        Imgproc.resize(faceMat, faceMat, new Size(100, 100));
        Mat faceHistogram = new Mat();
        computeHistogram(faceMat, faceHistogram);

        int bestMatchId = -1;
        double bestMatchValue = Double.MAX_VALUE;

        for (Map.Entry<Integer, Mat> entry : faceHistograms.entrySet()) {
            double matchValue = Imgproc.compareHist(faceHistogram, entry.getValue(), Imgproc.CV_COMP_CHISQR);
            if (matchValue < bestMatchValue) {
                bestMatchValue = matchValue;
                bestMatchId = entry.getKey();
            }
        }

        if (bestMatchValue > SUCCESS_THRESHOLD) {
            bestMatchId = -1;
        }

        return Map.entry(bestMatchId, bestMatchValue);
    }

    private void computeHistogram(Mat image, Mat histogram) {
        MatOfInt histSize = new MatOfInt(256);
        MatOfFloat histRange = new MatOfFloat(0f, 256f);
        MatOfInt channels = new MatOfInt(0);

        Imgproc.calcHist(Arrays.asList(image), channels, new Mat(), histogram, histSize, histRange);
        Core.normalize(histogram, histogram, 0, 1, Core.NORM_MINMAX);
    }

    private void updateAttendance(int studentId) {
        System.out.println("Sinh viên " + studentId + " đã được điểm danh.");
    }

    private void loadStudentData() {
        File datasetDir = new File(STUDENT_DATA_PATH);
        File[] studentDirs = datasetDir.listFiles(File::isDirectory);

        if (studentDirs != null) {
            for (File studentDir : studentDirs) {
                String studentName = studentDir.getName();
                String[] parts = studentName.split("_");

                try {
                    int studentId = Integer.parseInt(parts[0]);
                    studentNames.put(studentId, studentName);

                    File[] images = studentDir.listFiles((dir, name) -> name.endsWith(".jpg"));
                    if (images != null) {
                        Mat totalHistogram = new Mat();
                        for (File image : images) {
                            Mat img = Imgcodecs.imread(image.getAbsolutePath());
                            Mat faceHistogram = new Mat();
                            computeHistogram(img, faceHistogram);

                            if (totalHistogram.empty()) {
                                totalHistogram = faceHistogram.clone();
                            } else {
                                Core.add(totalHistogram, faceHistogram, totalHistogram);
                            }
                        }
                        Core.normalize(totalHistogram, totalHistogram, 0, 1, Core.NORM_MINMAX);
                        faceHistograms.put(studentId, totalHistogram);
                    }
                } catch (NumberFormatException e) {
                    System.out.println("Tên thư mục không hợp lệ: " + studentName);
                }
            }
        }
    }

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        Attendance attendance = new Attendance();
        attendance.startRecognition();
    }
}
