from gtts import gTTS

text = "heteroscedasticity"
tts = gTTS(text, lang='en')
tts.save("heteroscedasticity.mp3")
