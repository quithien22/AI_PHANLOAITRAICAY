import "./globals.css";

export const metadata = {
  title: "Phân loại trạng thái trái cây",
  description: "Ứng dụng AI nhận diện trái cây tươi hay hỏng",
};

export default function RootLayout({ children }) {
  return (
    <html lang="vi">
      <body>{children}</body>
    </html>
  );
}
