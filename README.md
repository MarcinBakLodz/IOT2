Ten plik zawiera aktualizacje stanu prac, uzupełnianą raz w tygodniu:

6.05.2025
VoiceCommands
	-naprawa dataset
 	-przerobienie poprzedniego modelu
  	-analiza funkcji strat

13.04.2025
dokończenie dataloadera i debugging
	-dodanie opcji zapisu i odczytu datasetów
	-generacja danych i augmentacja o noise
29.03.2025
zapoznanie się z pozyskanymi danymi
  	theory:
		-https://github.com/PyojinKim/Sensors-Data-Logger
		-https://developer.android.com/develop/sensors-and-location/sensors/sensors_overview
  	pratcice:
		-przejrzenie plików i stworzenie jednego pliku csv z oznaczeniami przejść, to pozwoli w przyszłości wykonać 4 testy:
			-tylko dane z kieszeni
			-tylko dane z nadgarstka
			-dane z obu miejsc na raz
			-dane z nadgarstka lub kieszeni (najcięższy przypadek, ale gdyby zadziałał najbardziej przydatny wtedy możnaby pomyśleć o modelu niewrażliwym na lokalizacje przy ciele i zawiera najwięcej danych)
zaplanowany typ sieci
	theory:
		-https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
		-https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
		-https://github.com/AlexanderVNikitin/tsgm/blob/main/tutorials/augmentations.ipynb
	practice:
		- TODO: stworzenie klasy dataloader, pozwalającej na określenie zaczytanych plików losowej chwili (w ten sposób możliwe będzie wygenerowanie wielu próbek z jednego przejscia). Domyślny format [batch, channel, lenght] 
		- TODO: dodanie do klasy metody augmentacji (szum, flipowanie danych w bok, Dynamic Time Warping Barycentric Average (DTWBA) żeby poznać coś nowego)
	
