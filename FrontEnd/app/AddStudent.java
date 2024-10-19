package app;

import java.io.File;

import org.opencv.core.Core;

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



	public static void main(String[] args) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

		launch(args);
	}
}
