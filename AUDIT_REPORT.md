# 📊 گزارش پیشرفت پروژه Robat - نسخه ۱.۰

---

## ۱. معماری فعلی سیستم

```
Robat/
├── data/                          # داده‌های تاریخی بازار
│   ├── BTCUSDT/5m                 # کندل‌های ۵ دقیقه‌ای
│   ├── ETHUSDT/5m
│   ├── SOLUSDT/5m
│   └── BTCUSDT_2025_mss_cache.parquet  # کش نتایج MSS
│
├── scripts/                       # اسکریپت‌های اجرا و تست
│   ├── download_data.py           # دانلود داده‌های بازار
│   ├── aggregate_data.py          # تجمیع داده‌ها
│   ├── test_mss.py                # تست موتور MSS با کش
│   ├── test_sweep.py              # تست شناسایی سوئیپ
│   ├── test_liquidity.py          # تست مناطق نقدینگی
│   └── test_structure.py          # تست ساختار بازار
│
├── src/
│   ├── config.py                  # تنظیمات مرکزی سیستم
│   ├── data/                      # لایه مدیریت داده
│   │   ├── downloader.py          # دانلود از صرافی
│   │   ├── storage.py             # ذخیره‌سازی و بارگذاری
│   │   ├── aggregator.py          # تبدیل تایم‌فریم‌ها
│   │   └── validator.py           # اعتبارسنجی داده‌ها
│   │
│   └── strategy/                  # موتور تحلیل تکنیکال
│       ├── swing_detector.py      # شناسایی سوینگ‌های داخلی
│       ├── liquidity.py           # شناسایی مناطق نقدینگی
│       ├── sweep_detector.py      # شناسایی رویدادهای سوئیپ
│       ├── mss_detector.py        # شناسایی Market Structure Shift
│       └── market_structure.py    # مدیریت ساختار بازار
```

### نقش `sweep_detector.py`:
شناسایی رویدادهای **Liquidity Sweep** شامل:
- Bullish Sweep (قیمت پایین‌تر از نقدینگی رفته و بالای آن بسته شود)
- Bearish Sweep (قیمت بالاتر از نقدینگی رفته و زیر آن بسته شود)
- محاسبه عمق سوئیپ (Depth) و طبقه‌بندی آن

### نقش `mss_detector.py`:
شناسایی **Market Structure Shift (MSS)** با الزامات:
- Bullish MSS: بسته شدن کندل بالای آخرین سوینگ‌های داخلی در ۱۲ کندل پس از سوئیپ
- Bearish MSS: بسته شدن کندل زیر آخرین سوینگ‌های داخلی در ۱۲ کندل پس از سوئیپ

---

## ۲. حسابرسی بهینه‌سازی - از O(N*M) به O(N log M)

### مشکل قبلی:
حلقه‌های تو در تو که برای هر سوئیپ، ۱۲ کندل بعدی را بررسی می‌کرد و برای هر کندل، تمام سوینگ‌های داخلی را جستجو می‌کرد:
- ۵۲۸,۵۷۰ سوئیپ × ۱۲ کندل × ۲,۴۳۶ سوینگ = **۱۵.۴ میلیارد عملیات**

### راه‌حل فعلی:

```python
# 1. پردازش سوینگ‌ها به DataFrame مرتب شده
highs_df = pd.DataFrame({
    "swing_confirmed": [s.confirmed_at for s in high_swings],
    "swing_high_price": [s.price for s in high_swings],
    ...
}).sort_values("swing_confirmed")

# 2. تولید برداری کاندیداها با numpy broadcasting
candle_indices = valid_sweep_idx[:, None] + offsets[None, :]

# 3. جستجوی برداری با merge_asof (O(log M))
cand_df = pd.merge_asof(
    cand_df, highs_df,
    left_on="timestamp", right_on="swing_confirmed",
    direction="backward"
)

# 4. اعمال شرایط به صورت برداری
bullish_mask = (
    (cand_df["sweep_direction"] == SweepDirection.BULLISH)
    & (cand_df["close"] > cand_df["swing_high_price"])
)
```

**مزایا:**
- حذف کامل حلقه‌های تو در تو
- استفاده از `searchsorted` برای جستجوی O(log N)
- `merge_asof` برای یافتن آخرین سوینگ تایید شده قبل از هر کندل

---

## ۳. رفع ایراد Look-Ahead Bias (مهر زمانی)

برای جلوگیری از استفاده از اطلاعات آینده، سیستم به شدت کنترل می‌کند:

```python
# در swing_detector.py
if confirmed_at <= current_time:
    candidates.append(swing)
```

**قوانین زمانی:**
- یک سوینگ تنها زمانی معتبر است که `confirmed_at ≤ زمان فعلی`
- در `detect_mss`، `merge_asof` با `direction="backward"` تضمین می‌کند که سوینگ‌هایی که **بعد** از زمان کندل تایید شده‌اند، هرگز استفاده نشوند
- این تضمین می‌کند که استراتژی در زمان واقعی نیز بدون خطا کار کند

---

## ۴. معیارهای عملکرد

| معیار | قبل از بهینه‌سازی | بعد از بهینه‌سازی |
|-------|------------------|------------------|
| پیچیدگی محاسباتی | O(N × 12 × M) = O(N·M) | O(N·12 + C·log M) |
| زمان اجرای بک‌تست ۲۰۲۵ | چند ساعت ( hanging ) | در حال اجرا |
| حافظه مصرفی | پایین | متوسط (DataFrame بزرگ) |
| تعداد عملیات | ۱۵.۴ میلیارد | ~۶ میلیون (کاندیداها) |

---

## ۵. نقشه راه فاز ۴ - سیستم امتیازدهی کنفلوئنس

### فاز ۴-الف: Displacement (+2 امتیاز)
```python
# شناسایی شکست قوی ساختار
def detect_displacement(df, mss_event):
    body_size = abs(close - open)
    avg_body = df.body.rolling(20).mean()
    if body_size > 1.5 * avg_body:
        return True
```

### فاز ۴-ب: OB/FVG Retests (+2 امتیاز)
```python
# شناسایی بازآزمایی منطقه عدم تعادل
def detect_ob_retest(df, mss_event):
    # اگر قیمت به منطقه Order Block بازگردد و از آن برگردد
    # بدون ایجاد ساختار جدید → +2 امتیاز
    pass
```

### سیستم امتیازدهی نهایی:
| رویداد | امتیاز | توضیح |
|--------|--------|-------|
| MSS استاندارد | +۱ | شناسایی اولیه |
| Displacement | +۲ | شکست قوی |
| OB Retest | +۲ | بازآزمایی تمیز |
| FVG Retest | +۲ | پر کردن خلأ |
| **حداقل ورود** | **+۴** | فیلتر معاملات |

---

## وضعیت فعلی سیستم

| ماژول | وضعیت | توضیحات |
|-------|-------|---------|
| `swing_detector.py` | ✅ کامل | شناسایی سوینگ‌های ۱h و ۴h |
| `liquidity.py` | ✅ کامل | مناطق نقدینگی |
| `sweep_detector.py` | ✅ کامل | شناسایی سوئیپ‌ها |
| `mss_detector.py` | ✅ بهینه‌سازی شده | نسخه برداری |
| `test_mss.py` | 🔄 در حال اجرا | کش Parquet |
| سیستم امتیازدهی | ⏳ در انتظار | فاز ۴ |

---

> **یادداشت:** اسکریپت `test_mss.py` در زمان نگارش این گزارش در حال اجراست و نتایج نهایی پس از تکمیل محاسبات در فایل `data/BTCUSDT_2025_mss_cache.parquet` ذخیره خواهد شد.
