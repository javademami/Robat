"""
Generate robot report file.
"""

from pathlib import Path

REPORT = """# 📊 گزارش کامل فنی ربات «تک‌تیرانداز»

**نام:** تک‌تیرانداز (Tak-Tirandaz)
**نسخه:** V1.0
**استراتژی:** ICT/SMC
**تاریخ:** ۲۰۲۶

---

## 🎯 خلاصه مدیریتی

| شاخص | نتیجه |
|-------|--------|
| استراتژی | ICT/SMC |
| تعداد نماد | ۴ |
| دوره بک‌تست | ۱۰ سال |
| تعداد معاملات | ۳۶۲ |
| Win Rate | 35.64% |
| Total R | +154 |
| Profit Factor | 1.66 |
| Max Drawdown | 26 R |
| وضعیت | ✅ آماده |

---

## 🔧 مکانیزم‌های تشخیص (۱۰ فیلتر)

### ۱. Liquidity Zones
- Equal Highs/Lows
- PDH/PDL
- Swing High/Low

### ۲. Liquidity Sweep
- شکست سطح + برگشت سریع

### ۳. MSS
- شکست ساختار بعد از Sweep

### ۴. Displacement
- Body/ATR >= 1.0
- CLV مناسب

### ۵. Order Block
- آخرین کندل مخالف قبل از Displacement

### ۶. FVG
- شکاف بین کندل‌ها

### ۷. Retest
- برگشت به Zone

### ۸. Micro-MSS
- شکست ساختار ریز

### ۹. HTF Bias
- جهت‌گیری ۴H

### ۱۰. Entry Score
- ۱۰۰ امتیاز

---

## 💰 مدیریت ریسک

| پارامتر | مقدار |
|---------|-------|
| ریسک هر معامله | ۱٪ |
| RR | 3.0 |
| SL Buffer | 0.10 ATR |
| Max Positions | ۳ |

---

## 📊 نتایج ۱۰ ساله

| معیار | مقدار |
|-------|-------|
| معاملات | ۳۶۲ |
| Win Rate | 35.64% |
| Total R | +154 |
| PF | 1.66 |

---

## 🏁 جمع‌بندی

ربات تک‌تیرانداز آماده Paper Trading است.

**امضا:** تیم توسعه
**تاریخ:** ۲۰۲۶
"""

# Save
output_path = Path(r"C:\Users\EnRival\Robat\data\validation\robot_report_full.md")
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(REPORT, encoding="utf-8")

print(f"Saved: {output_path}")