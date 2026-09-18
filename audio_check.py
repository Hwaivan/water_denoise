import soundfile as sf

# path = (
#     "/data/huayifan/water_denoise/data/"
#     "ShipsEar-12class_W5H1/test/snr_-5dB/"
#     "000000_Dredger_93__A__Draga_1_0000_0.00s-5.00s.wav"
# )

path = '/data/huayifan/water_denoise/logs/dccrn_snr_m10_10/evaluation_best/audio_examples/0000_enhanced.wav'

info = sf.info(path)

print("采样率：", info.samplerate)
print("通道数：", info.channels)
print("采样点数：", info.frames)
print("时长：", info.duration)
print("格式：", info.format)
print("编码：", info.subtype)