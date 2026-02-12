# 🎯 PPE Violation Alert Testing Guide

## What You're Testing
You said you **won't be wearing a vest**, so the system should detect a **NO_VEST violation** after 10 continuous seconds.

---

## 📹 What You'll See in the Webcam Window

### Visual Elements:
1. **Green Box Around You** - "Person ID: X" (always green)
2. **Red "No Vest" Box** - Around your torso area (missing vest indicator)
3. **Helmet Box** - Yellow/Green if wearing helmet
4. **FPS & Track Count** - Top left corner

### Example:
```
┌─────────────────────────────────────┐
│ FPS: 12.5 | Tracks: 1               │
│                                     │
│  ┌─────────────────┐                │
│  │   Person ID: 1  │ (Green Box)   │
│  │  ┌──────────┐   │                │
│  │  │Helmet 0.8│   │ (Yellow Box)  │
│  │  └──────────┘   │                │
│  │  ┌──────────┐   │                │
│  │  │ No Vest  │   │ (RED Box!)    │
│  │  └──────────┘   │                │
│  └─────────────────┘                │
└─────────────────────────────────────┘
```

---

## ⏱️ Timeline - What Happens

### **0s - 10s** (First 10 seconds)
- You appear on camera
- Red "No Vest" box appears
- System is tracking but **NOT alerting yet**
- Watch the webcam terminal...

### **At 10s** (Violation threshold met!)
You'll see this in the **webcam terminal**:
```
WARNING - 🚨 PPE VIOLATION ALERT - Track 1 - NO_VEST - Camera: webcam - Time: 2026-02-01T22:14:30Z
```

### **After 10s** (Alert sent!)
- **NO duplicate alerts** (even if you stay without vest)
- Alert is now stored in the API
- You can view it in your browser

---

## 🌐 How to View the Alert

### **Option 1: Browser** (Recommended)
1. Open your browser
2. Go to: **http://localhost:8000/api/ppe-alerts**
3. You'll see JSON like this:
```json
{
  "total": 1,
  "alerts": [
    {
      "track_id": 1,
      "timestamp": "2026-02-01T22:14:30Z",
      "camera_id": "webcam",
      "violation_type": "NO_VEST"
    }
  ]
}
```

### **Option 2: API Docs** (Interactive)
1. Go to: **http://localhost:8000/docs**
2. Scroll to "GET /api/ppe-alerts"
3. Click "Try it out" → "Execute"
4. See the response below

### **Option 3: Stats Page**
Go to: **http://localhost:8000/api/ppe-alerts/stats**
```json
{
  "total_alerts": 1,
  "NO_HELMET": 0,
  "NO_VEST": 1,
  "NO_BOTH": 0
}
```

---

## 🧪 Test Scenarios

### **Scenario 1: NO_VEST (What you're doing)**
1. ✅ Stand without vest (but with helmet if you have one)
2. ⏱️ Wait 10 seconds
3. 🚨 Alert fires: "NO_VEST"
4. 🌐 Check browser: http://localhost:8000/api/ppe-alerts

### **Scenario 2: Re-Alert Test**
1. 👕 Put on a vest (become compliant)
2. ✅ Red "No Vest" box disappears
3. 👕 Remove vest again
4. ⏱️ Wait another 10 seconds
5. 🚨 **NEW alert fires!** (Re-alert capability)

### **Scenario 3: No Duplicate Test**
1. ⏱️ Get first alert at 10s
2. ⏱️ Stay without vest for another 20 seconds (total 30s)
3. ✅ **NO new alerts** (duplicate prevention working!)

---

## 🔍 Where to Look

### **Webcam Terminal** (Real-time logs)
Watch for:
```
🚨 PPE VIOLATION ALERT - Track X - NO_VEST
```

### **API Server Terminal** (If running separately)
Same message appears there too

### **Browser**
- **http://localhost:8000/api/ppe-alerts** - See all alerts
- **http://localhost:8000/api/ppe-alerts/stats** - See counts
- **http://localhost:8000/docs** - Interactive API

---

## ⚙️ Current Configuration

Your system is configured with:
- **Violation Threshold**: 10.0 seconds
- **Monitoring**: Enabled ✅
- **Alert Endpoint**: http://localhost:8000/api/ppe-alert

---

## 🎬 Ready to Test!

Your webcam should now be running. 

**What to do:**
1. ✅ Stand in front of camera (without vest)
2. ⏱️ Count to 10 slowly
3. 👀 Watch the webcam terminal for the alert
4. 🌐 Open browser: http://localhost:8000/api/ppe-alerts
5. 📊 See your violation recorded!

**Press 'q' in the webcam window when done testing.**
