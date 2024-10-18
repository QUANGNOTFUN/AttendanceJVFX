package app;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;

public class MainApp extends Application {

	@Override
	public void start(Stage primaryStage) {
		// Tạo các nút
		Button diemDanhButton = new Button("Điểm danh");
		Button adminButton = new Button("Admin");
		Button thoatButton = new Button("Thoát");

		// Đặt hành động cho nút "Điểm danh" để mở cửa sổ Attendance
		diemDanhButton.setOnAction(e -> {
			Attendance attendance = new Attendance();
			Stage attendanceStage = new Stage(); // Tạo cửa sổ mới
			attendance.startRecognition(); // Mở Attendance trong cửa sổ mới

			// Đặt tiêu đề và kích thước cho cửa sổ Attendance
			attendanceStage.setTitle("Điểm danh");
			attendanceStage.setWidth(800); // Thiết lập chiều rộng
			attendanceStage.setHeight(600); // Thiết lập chiều cao
			attendanceStage.show(); // Hiện thị cửa sổ mới
		});

		// Đặt hành động cho nút "Admin" để mở cửa sổ Admin
		adminButton.setOnAction(e -> {
			Admin admin = new Admin();
			Stage adminStage = new Stage(); // Tạo cửa sổ mới cho Admin
			admin.start(adminStage); // Mở Admin trong cửa sổ mới

			// Đặt tiêu đề cho cửa sổ Admin
			adminStage.setTitle("Admin");
			adminStage.show(); // Hiện thị cửa sổ mới
		});

		// Đặt hành động cho nút "Thoát" để thoát ứng dụng
		thoatButton.setOnAction(e -> {
			System.out.println("Thoát ứng dụng"); // In thông báo khi thoát
			primaryStage.close(); // Đóng cửa sổ ứng dụng chính
		});

		// Tạo VBox để chứa các nút
		VBox layout = new VBox(10); // Khoảng cách giữa các nút là 10
		layout.getChildren().addAll(diemDanhButton, adminButton, thoatButton);

		// Đặt style cho layout
		layout.setStyle("-fx-padding: 20; -fx-alignment: center;");

		// Tạo scene và thiết lập cho stage
		Scene scene = new Scene(layout, 300, 200);
		scene.getStylesheets().add(getClass().getResource("style.css").toExternalForm()); // Liên kết đến CSS
		primaryStage.setScene(scene);
		primaryStage.setTitle("Ứng dụng Điểm danh");
		primaryStage.show();
	}

	public static void main(String[] args) {
		launch(args);
	}
}
