ROOT = "..\\datasets\\"

IMBALANCED_DATASETS = {
#    "BreastCancer": "UCI_BreastCancer.arff",
    "Climate": "UCI_Climate.arff",
    "Ecoli3": "Keel_Ecoli_3.arff",
    "Ionosphere": "UCI_Ionosphere.arff",
    "Cleveland0v4": "Keel_Cleveland_0vs4.arff",
#    "Pima": "UCI_Pima.arff",
    "Vehicle1": "Keel_Vehicle_1.arff",
    "Yeast2v4": "Keel_Yeast_2vs4.arff",
	"Abalone9v21": "Keel_Abalone_9vs21.arff",
	"Dermatology6": "Keel_Dermatology_6.arff",
#	"Ecoli0v1": "Keel_Ecoli_0vs1.arff",
#	"Glass1": "Keel_Glass_1.arff",
	"Glass6": "Keel_Glass_6.arff",
#	"Iris0": "Keel_Iris0.arff",
	"LED7digit": "Keel_Led7digit_Allvs1.arff",
	"Poker9v7": "Keel_Poker_9vs7.arff",
	"Shuttle6v23": "Keel_Shuttle_6vs23.arff",
	"Vehicle3": "Keel_Vehicle_3.arff",
	"Vowel0": "Keel_Vowel_0.arff",
 	"WineQualityRed8v6": "Keel_Winequality_red_8vs6.arff",
 	"WineQualityRed4": "Keel_winequality-red-4.arff",
	"Yeast1v7": "Keel_yeast-1_vs_7.arff",
#	"Banknote": "UCI_Banknote.arff",
	"Parkinsons": "UCI_Parkinsons.arff",
	"Relax": "UCI_Relax.arff",
	"Transfusion": "UCI_Transfusion.arff",
}

HEAVILY_IMBALANCED = {
    "Yeast2v4": "Keel_Yeast_2vs4.arff",
	"Cleveland0v4": "Keel_Cleveland_0vs4.arff",
	"Dermatology6": "Keel_Dermatology_6.arff",
	"Abalone9v21": "Keel_Abalone_9vs21.arff",
	"Ecoli3": "Keel_Ecoli_3.arff",
	"Glass6": "Keel_Glass_6.arff",
	"LED7digit": "Keel_Led7digit_Allvs1.arff",
	"Poker9v7": "Keel_Poker_9vs7.arff",
	"Shuttle6v23": "Keel_Shuttle_6vs23.arff",
	"Vowel0": "Keel_Vowel_0.arff",
	"WineQualityRed8v6": "Keel_Winequality_red_8vs6.arff",
	"Climate": "UCI_Climate.arff",
	"WineQualityRed4": "Keel_winequality-red-4.arff",
	"Yeast1v7": "Keel_yeast-1_vs_7.arff",
}

SYNTHETIC_DATASETS = {
	"Paw_600_5_0_BI": "Keel_synth_paw02a_600_5_0-BI.arff",
    # "Subcl_600_5_30_BI": "Keel_synth_03subcl5-600-5-30-BI.arff",
    # "Clover_600_5_30_BI": "Keel_synth_04clover5z-600-5-30-BI.arff",
    "Paw_600_5_30_BI": "Keel_synth_paw02a-600-5-30-BI.arff",
}

SDP = {
	"Cm1": "Cm1.csv",
	"Jm1": "Jm1.csv",
	"Kc1": "Kc1.csv",
	"Kc3": "Kc3.csv",
	"Mc1": "Mc1.csv",
	"Mc2": "Mc2.csv",
	"Mw1": "Mw1.csv",
	"Pc1": "Pc1.csv",
	"Pc2": "Pc2.csv",
	"Pc3": "Pc3.csv",
	"Pc4": "Pc4.csv"
}

DS_PhD = {
	"Sonar": "Keel_Sonar.arff",
	"Spect": "spect.arff",
	"BreastCancer": "breastcancer.arff",
	"Ionosphere": "UCI_Ionosphere.arff",
	"Biodegradeable": "UCI_Biodeg.arff",
	"Voice": "UCI_Voice.arff",
	"GermanCredit": "german.arff",
	"Parkinsons": "UCI_Parkinsons.arff",
	"Clean2": "clean2.arff",
	"Ecoli3": "Keel_Ecoli_3.arff",
	"Yeast2v4": "Keel_Yeast_2vs4.arff",
	"Climate": "UCI_Climate.arff",
	"LED7digit": "Keel_Led7digit_Allvs1.arff",
	"Shuttle6v23": "Keel_Shuttle_6vs23.arff",
	"Poker6v23": "Keel_Poker_6v23.arff",
	"kddcup-guess_passwd_vs_satan": "kddcup-guess_passwd_vs_satan.arff",
	"Abalone21v8": "Keel_Abalone_9v21.arff",
	"WineQualityWhite3v7": "winequality-white-3_vs_7.arff"
}

DS_PhD_Processed = {
	"Sonar": "Sonar.csv",
	"Spect": "Spect.csv",
	"BreastCancer": "Breastcancer.csv",
	"Ionosphere": "Ionosphere.csv",
	"Biodegradeable": "Biodegradeable.csv",
	"Voice": "Uci_voice.csv",
	"GermanCredit": "German_credit.csv",
	"Parkinsons": "Uci_parkinsons.csv",
	# "Clean2": "Clean2.csv",
	"Hepatitis": "Hepatitis.csv",
	"Ecoli3": "Ecoli3.csv",
	"Yeast2v4": "Yeast_2vs4.csv",
	"Climate": "Climate.csv",
	"LED7digit": "LED7digit_allvs1.csv",
	"Shuttle6v23": "Shuttle_6vs23.csv",
	"Poker9v27": "Poker-9_vs_7.csv",
	"Kddcup-guess_passwd_vs_satan": "Kddcup-guess_passwd_vs_satan.csv",
	"Abalone21v8": "Abalone-21_vs_8.csv",
	"WineQualityWhite3v7": "Keel_winequality_red_8vs6.csv"
}