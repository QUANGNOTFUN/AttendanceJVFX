package app;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;

public class Admin extends Application {

	@Override
	public void start(Stage primaryStage) {
		// Tạo các nút cho giao diện Admin
		Button quanLySinhVienButton = new Button("Thêm sinh viên");
		Button quanLyDiemDanhButton = new Button("Quản lý Điểm danh");
		Button quayLaiButton = new Button("Quay lại");

		// Đặt hành động cho nút Quản lý Sinh viên
		quanLySinhVienButton.setOnAction(e -> {
			AddStudent addStudent = new AddStudent();
			Stage addStudentStage = new Stage(); // Tạo cửa sổ mới cho Thêm sinh viên
			addStudent.start(addStudentStage); // Mở cửa sổ Thêm sinh viên
		});

		// Đặt hành động cho nút Quản lý Điểm danh
		quanLyDiemDanhButton.setOnAction(e -> {
			System.out.println("Quản lý Điểm danh đã được chọn");
			// Thêm mã để mở giao diện quản lý điểm danh (nếu có)
		});

		// Đặt hành động cho nút Quay lại để quay về màn hình chính (Main)
		quayLaiButton.setOnAction(e -> {
			System.out.println("Quay lại màn hình chính");
			primaryStage.close(); // Đóng cửa sổ Admin
		});

		// Tạo VBox để chứa các nút
		VBox layout = new VBox(10); // Khoảng cách giữa các nút là 10
		layout.getChildren().addAll(quanLySinhVienButton, quanLyDiemDanhButton, quayLaiButton);

		// Đặt style cho layout
		layout.setStyle("-fx-padding: 20; -fx-alignment: center;");

		// Tạo cảnh và thiết lập nó vào giai đoạn
		Scene scene = new Scene(layout, 300, 200);
		primaryStage.setScene(scene);
		primaryStage.setTitle("Admin Panel");
		primaryStage.show();
	}

	public static void main(String[] args) {
		launch(args);
	}
}
