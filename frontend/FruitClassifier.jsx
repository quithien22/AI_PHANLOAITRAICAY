"use client"; // Cần cho Next.js App Router; React thuần (Vite/CRA) bỏ qua dòng này cũng được.

import { useEffect, useRef, useState } from "react";
import "./FruitClassifier.css";

const API_URL = "http://localhost:8000/predict";

// Nhãn từ backend (không phân biệt hoa thường). Hiện model có 2 nhãn: fresh, rotten.
// unripe/ripe giữ lại để dùng khi bạn mở rộng lên 3 nhãn.
const LABELS = {
  fresh: { text: "Tươi", note: "Trái cây còn tươi, có thể ăn được.", tone: "unripe" },
  unripe: { text: "Xanh", note: "Trái cây còn xanh, chưa nên ăn. Để thêm vài ngày.", tone: "unripe" },
  ripe: { text: "Chín", note: "Trái cây đã chín, ăn ngon nhất lúc này.", tone: "ripe" },
  rotten: { text: "Hư hỏng", note: "Trái cây đã hỏng, không nên ăn.", tone: "rotten" },
};

export default function FruitClassifier() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  // Giải phóng URL tạm khi đổi ảnh hoặc rời trang
  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  const chooseFile = (selected) => {
    if (!selected) return;
    if (!selected.type.startsWith("image/")) {
      setError("Vui lòng chọn một file ảnh (JPG, PNG, WEBP).");
      return;
    }
    setError("");
    setResult(null);
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    chooseFile(e.dataTransfer.files?.[0]);
  };

  const handleReset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError("");
    if (inputRef.current) inputRef.current.value = "";
  };

  const handleAnalyze = async () => {
    if (!file || loading) return;
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(API_URL, { method: "POST", body: formData });
      const data = await res.json().catch(() => ({}));

      if (!res.ok) {
        throw new Error(data.detail || `Máy chủ trả về lỗi ${res.status}.`);
      }
      setResult(data);
    } catch (err) {
      setError(
        err instanceof TypeError
          ? "Không kết nối được máy chủ. Hãy kiểm tra backend đã chạy ở cổng 8000 chưa."
          : err.message
      );
    } finally {
      setLoading(false);
    }
  };

  const info = result ? LABELS[String(result.label).toLowerCase()] : null;

  return (
    <main className="fc-page">
      <section className="fc-card">
        <header className="fc-head">
          <h1>Kiểm tra độ chín của trái cây</h1>
          <p>Tải ảnh lên, hệ thống sẽ cho biết trái cây còn tươi hay đã hỏng.</p>
        </header>

        <div className="fc-body">
          <div
            className={`fc-drop ${dragging ? "is-dragging" : ""} ${preview ? "has-image" : ""}`}
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
            aria-label="Chọn ảnh trái cây"
          >
            {preview ? (
              <img src={preview} alt="Ảnh trái cây đã chọn" className="fc-preview" />
            ) : (
              <div className="fc-placeholder">
                <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
                  <path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5" strokeLinecap="round" strokeLinejoin="round" />
                  <path d="M4 15v3a2 2 0 002 2h12a2 2 0 002-2v-3" strokeLinecap="round" />
                </svg>
                <strong>Kéo thả ảnh vào đây</strong>
                <span>hoặc bấm để chọn từ máy</span>
              </div>
            )}
            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              hidden
              onChange={(e) => chooseFile(e.target.files?.[0])}
            />
          </div>

          <div className="fc-side">
            <div className="fc-actions">
              <button className="fc-btn fc-btn-primary" onClick={handleAnalyze} disabled={!file || loading}>
                {loading ? (
                  <>
                    <span className="fc-spinner" aria-hidden="true" /> Đang phân tích...
                  </>
                ) : (
                  "Phân tích"
                )}
              </button>
              {file && !loading && (
                <button className="fc-btn fc-btn-ghost" onClick={handleReset}>
                  Chọn ảnh khác
                </button>
              )}
            </div>

            <div className="fc-result-area" aria-live="polite">
              {error && <div className="fc-error">{error}</div>}

              {!error && !result && !loading && (
                <p className="fc-hint">
                  {file ? "Bấm “Phân tích” để xem kết quả." : "Kết quả sẽ hiện ở đây."}
                </p>
              )}

              {info && (
                <div className={`fc-result tone-${info.tone}`}>
                  <span className="fc-result-label">Trạng thái</span>
                  <span className="fc-result-value">{info.text}</span>

                  <div className="fc-meter-head">
                    <span>Độ tin cậy</span>
                    <strong>{Number(result.confidence).toFixed(1)}%</strong>
                  </div>
                  <div className="fc-meter" role="progressbar" aria-valuenow={result.confidence} aria-valuemin={0} aria-valuemax={100}>
                    <div className="fc-meter-fill" style={{ width: `${result.confidence}%` }} />
                  </div>

                  <p className="fc-note">{info.note}</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}
