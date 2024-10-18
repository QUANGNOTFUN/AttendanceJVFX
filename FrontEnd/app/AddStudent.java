package app;

import java.io.File;

import connect.ConnectSQL;
import controller.FaceDetect;
import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.layout.GridPane;
import javafx.stage.Stage;
import model.Student;

public class AddStudent extends Application {

	private TextField nameField;
	private TextField classField;
	private FaceDetect faceDetect; // Đối tượng FaceDetect
	private ConnectSQL connectSQL;

	@Override
	public void start(Stage primaryStage) {
		// Tạo các trường nhập liệu
		nameField = new TextField();
		classField = new TextField();

		// Tạo nút thêm sinh viên
		Button addButton = new Button("Thêm sinh viên");
		addButton.setOnAction(e -> addStudent());

		// Tạo giao diện
		GridPane grid = new GridPane();
		grid.setHgap(10);
		grid.setVgap(10);
		grid.add(new Label("Tên sinh viên:"), 0, 0);
		grid.add(nameField, 1, 0);
		grid.add(new Label("Lớp sinh viên:"), 0, 1);
		grid.add(classField, 1, 1);
		grid.add(addButton, 1, 2);

		Scene scene = new Scene(grid, 300, 200);
		primaryStage.setScene(scene);
		primaryStage.setTitle("Thêm Sinh Viên");
		primaryStage.show();
	}

	private void addStudent() {
		try {
			// Nhập thông tin sinh viên từ các trường
			String name = nameField.getText();
			String studentClass = classField.getText();

			// Kết nối đến cơ sở dữ liệu để lấy ID tiếp theo
			connectSQL = new ConnectSQL();
			if (connectSQL.connect_db()) {
				// Tạo đối tượng Student với ID tạm thời
				Student student = new Student(0, name, studentClass, null); // ID sẽ tự động sinh từ database

				// Lấy ID mới từ cơ sở dữ liệu
				int newId = connectSQL.getNextStudentId(); // Lấy ID mới từ cơ sở dữ liệu
				student.setId(newId); // Gán ID mới cho sinh viên

				// Tạo thư mục để lưu ảnh khuôn mặt
				String studentDirectory = "dataset/" + newId; // Tạo đường dẫn thư mục từ ID
				File directory = new File(studentDirectory);
				if (!directory.exists()) {
					directory.mkdirs(); // Tạo thư mục nếu chưa tồn tại
				}

				// Khởi tạo và chạy FaceDetect để chụp ảnh
				faceDetect = new FaceDetect();
				faceDetect.detectFace(student);

				// Lấy đường dẫn ảnh đã lưu trong đối tượng student
				String imagePath = student.getFaceImgPath();
				if (imagePath != null) {
					// Lưu thông tin sinh viên vào cơ sở dữ liệu
					connectSQL.addStudentToDatabase(newId, name, studentClass, imagePath);

					// Kiểm tra trạng thái nhận diện
					checkAttendance(student);

					// Hiển thị thông tin sinh viên
					student.displayStudentInfo();
				}

				// Làm sạch các trường nhập liệu sau khi thêm
				nameField.clear();
				classField.clear();
			}
		} catch (Exception e) {
			System.out.println("Có lỗi xảy ra: " + e.getMessage());
		}
	}

	private void checkAttendance(Student student) {
		new Thread(() -> {
			int presentCount = 0;
			long startTime = System.currentTimeMillis();

			while (true) {
				double accuracy = faceDetect.getRecognitionAccuracy(); // Giả sử có phương thức này
				if (accuracy > 70) {
					presentCount++;
				} else {
					presentCount = 0; // Reset nếu không đạt
				}

				// Kiểm tra nếu đã đạt 5 giây
				if (presentCount >= 5) {
					connectSQL.updateStudentStatus(student.getId(), "present");
					break; // Thoát khỏi vòng lặp
				}

				// Delay 1 giây trước khi kiểm tra lại
				try {
					Thread.sleep(1000);
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
		}).start();
	}

	public static void main(String[] args) {
		launch(args);
	}
}
