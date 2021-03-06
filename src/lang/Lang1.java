package lang;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;
/*
Single layer implementation-no optimizers
*/
public class Lang1{
	static double step = .005;
	static final int n_neurons = 20; //number of neurons in hidden layer
	static final int char_max = 15; //maximum length of word
	static boolean speed = true; //print out advanced data (word + individ. weights for each language + prediction)
	static boolean acc = true; //print out accuracy in training
	static boolean count = true; //print out # of iter'ns remaining in training
	static double a = 1; //var. stores accuracy; init'd to 1
	static double initW = 1; //initial weight
	//chi,jap,eng,spn,kor,fre,ger,rus,vie,swe,arb,hin,bng,ita
	// 0   1   2   3   4   5   6   7   8   9  10  11  12  13 
	static int[] testing = {1,2}; //network trains on selected languages
	
	static double weight1[][][] = new double[char_max][27][n_neurons];
	static double weight2[][] = new double[n_neurons][14];
	
	static double in1[][] = new double[char_max][27];
	static double in2[] = new double[n_neurons];
	static double out[] = new double[14];
	//stores lang_data
	static ArrayList<String> g1 = new ArrayList<String>(),
			g2 = new ArrayList<String>(),
			g3 = new ArrayList<String>(), 
			g4 = new ArrayList<String>(),
			g5 = new ArrayList<String>(),
			g6 = new ArrayList<String>(),
			g7 = new ArrayList<String>(),
			g8 = new ArrayList<String>(),
			g9 = new ArrayList<String>(),
			g10 = new ArrayList<String>(),
			g11 = new ArrayList<String>(),
			g12 = new ArrayList<String>(),
			g13 = new ArrayList<String>(),
			g14 = new ArrayList<String>();
	
	public static ArrayList<Boolean> bool = new ArrayList<Boolean>();
	
	/*
	 * returns double rounded to 3 decimal points
	 */
	public static double r(double x) {
		return (int)(x * 1000)/1000.0;
	}
	public static double sigmoid(double x) {
		if(x < -8){
		    return 0;
		}else if(x > 8){
		    return 1;
		}else {
		    return 1 / (1 + Math.exp(-x));
		}
	}
	
	/*
	 * train network for i iterations
	 */
	public static void train(int i) {
		while(i > 0) {
			if(count)System.out.println(i);
			int group = (int)(Math.random() * testing.length);
			group = testing[group];
			
			int idx;
			switch(group) {
			case 0: idx = (int)(Math.random() * g1.size());
					adj(g1.get(idx), group);
					break;
			case 1: idx = (int)(Math.random() * g2.size());
					adj(g2.get(idx), group);
					break;
			case 2: idx = (int)(Math.random() * g3.size());
					adj(g3.get(idx), group);
					break;
			case 3: idx = (int)(Math.random() * g4.size());
					adj(g4.get(idx), group);
					break;
			case 4: idx = (int)(Math.random() * g5.size());
					adj(g5.get(idx), group);
					break;
			case 5: idx = (int)(Math.random() * g6.size());
					adj(g6.get(idx), group);
						break;
			case 6: idx = (int)(Math.random() * g7.size());
					adj(g7.get(idx), group);
					break;
			case 7: idx = (int)(Math.random() * g8.size());
					adj(g8.get(idx), group);
					break;
			case 8: idx = (int)(Math.random() * g9.size());
					adj(g9.get(idx), group);
					break;
			case 9: idx = (int)(Math.random() * g10.size());
					adj(g10.get(idx), group);
					break;
			case 10: idx = (int)(Math.random() * g11.size());
					adj(g11.get(idx), group);
					break;
			case 11: idx = (int)(Math.random() * g12.size());
					adj(g12.get(idx), group);
					break;
			case 12: idx = (int)(Math.random() * g13.size());
					adj(g13.get(idx), group);
					break;
			case 13: idx = (int)(Math.random() * g14.size());
					adj(g14.get(idx), group);
					break;
			case 14:  
			default:System.out.println("how."); break;
			}
			i--;
		}
	}
	
	/*
	 * inputs String s into neural network
	 */
	public static void input(String s) {
		s += ".";
		in1 = new double[char_max][27];
		for(int i = 0; i < in1.length; i++) {
			for(int j = 0; j < in1[0].length; j++) {
				in1[i][j] = 0;
			}
		}
		for(int i = 0; i < in1.length; i++) {
			int v = s.charAt(i) - 'a';
			if(v < 0 || v > 'z' - 'a') {
				while(i < in1.length) {
					in1[i][26] = 1;
					i++;
				}
			}else {
				in1[i][v] = 1;
			}
		}
	}
	
	/*
	 * passes String s through neural network
	 */
	public static void test(String s) {
		input(s);
		for(int i = 0; i < in2.length; i++) {
			double sum = 0;
			for(int j = 0; j < in1.length; j++) {
				for(int k = 0; k < in1[0].length; k++) {
					sum += weight1[j][k][i] * in1[j][k];
				}
			}
			in2[i] = sigmoid(sum);
		}
		for(int i = 0; i < out.length; i++) {
			double sum = 0;
			for(int j = 0; j < in2.length; j++) {
				sum += weight2[j][i] * in2[j];
			}
			out[i] = sigmoid(sum);
		}
	}
	
	/*
	 * passses String s and adjusts weights of neural network
	 */
	public static void adj(String s, int goal) {
		test(s);
		
		int idx = 0;
		for(int i = 1; i < out.length; i++) {if(out[i] > out[idx]) {idx = i;}}
		if(!speed) {
			System.out.print(s + ";" + goal + "[" + r(out[0]));
			for(int i = 1; i < out.length; i++) {System.out.print("," + r(out[i]));}
			System.out.print("]");
			System.out.print("Prediction:" );
			switch(idx) {
			case 0: System.out.print("MANDARIN"); break;
			case 1: System.out.print("JAPANESE"); break;
			case 2: System.out.print("ENGLISH"); break;
			case 3: System.out.print("SPANISH"); break;
			case 4: System.out.print("KOREAN"); break;
			case 5: System.out.print("FRENCH"); break;
			case 6: System.out.print("GERMAN"); break;
			case 7: System.out.print("RUSSIAN"); break;
			case 8: System.out.print("VIETNAMESE"); break;
			case 9: System.out.print("SWEDISH"); break;
			case 10: System.out.print("ARABIC"); break;
			case 11: System.out.print("HINDI"); break;
			case 12: System.out.print("BENGALI"); break;
			case 13: System.out.print("ITALIAN"); break;
			}
		}
		if(acc) {
			System.out.print("(");
			if(goal == idx) {
				System.out.print("CORRECT!");
				bool.add(true);
			}else {
				System.out.print("WRONG");
				bool.add(false);
			}
			if(bool.size() > 10000) {
				bool.remove(0);
			}
			double ct = 0;
			for(int i = 0 ; i < bool.size(); i++ ) {
				if(bool.get(i)) {ct++;}
			}
			System.out.print(")");
			a = ct / bool.size();
			System.out.println(" ACC:" + ct / bool.size());
			
		}
		for(int i = 0; i < out.length; i++) {
			int t = 0;
			if(goal == i) {t = 1;}
			for(int j = 0; j < in2.length; j++) {
				weight2[j][i] -= step * (out[i] - t) * out[i] * (1 - out[i]) * in2[j];
			}
		}
		double[] ta = new double[in2.length];
		for(int i = 0; i < out.length; i++) {
			int t = 0;
			if(goal == i) {t = 1;}
			for(int j = 0; j < in2.length; j++) {
				ta[j] += (out[i] - t) * out[i] * (1 - out[i]) * weight2[j][i];
			}
		}
		for(int i = 0; i < in2.length; i++) {
			for(int j = 0; j < in1.length; j++) {
				for(int k = 0; k < in1[0].length; k++) {
					if(in1[j][k] == 0) {continue;}
				//	if(i == 0)System.out.println(ta[i]); //ALL TARGETS ARE ZERO
					weight1[j][k][i] -= step * (ta[i]) * in2[i] * (1 - in2[i]) * in1[j][k];
				}
			}
		}
	}
	
	/*
	 * initializes lists, network
	 */
	public static void main(String[] args) throws IOException {
		for(int i = 0; i < weight1.length; i++) {
			for(int j = 0; j < weight1[0].length; j++) {
				for(int k = 0; k < weight1[0][0].length; k++) {
					weight1[i][j][k] = (Math.random() * initW) - (initW / 2);
				}
			}
		}
		for(int i = 0; i < weight2.length; i++) {
			for(int j = 0; j < weight2[0].length; j++) {
				weight2[i][j] = (Math.random() * initW) - (initW / 2);
			}
		}
		
		@SuppressWarnings("resource")
		BufferedReader f = new BufferedReader(new FileReader("src/data/train_chi.txt"));
		System.out.print("Loading");
		while(f.ready()) {
			g1.add(f.readLine());
		}
		System.out.print(".");
		f = new BufferedReader(new FileReader("src/data/train_jap.txt"));
		while(f.ready()) {
			g2.add(f.readLine());
		}
		System.out.print(".");
		f = new BufferedReader(new FileReader("src/data/train_eng.txt"));
		while(f.ready()) {
			g3.add(f.readLine().toLowerCase());
		}
		System.out.print(".");
		f = new BufferedReader(new FileReader("src/data/train_spn.txt"));
		while(f.ready()) {
			g4.add(f.readLine());
		}
		System.out.print(".");
		f = new BufferedReader(new FileReader("src/data/train_kor.txt"));
		while(f.ready()) {
			g5.add(f.readLine());
		}
		System.out.print(".");
		f = new BufferedReader(new FileReader("src/data/train_fre.txt"));
		while(f.ready()) {
			g6.add(f.readLine());
		}
		System.out.print(".");
		f = new BufferedReader(new FileReader("src/data/train_ger.txt"));
		while(f.ready()) {
			g7.add(f.readLine());
		}
		System.out.print(".");
		f = new BufferedReader(new FileReader("src/data/train_rus.txt"));
		while(f.ready()) {
			g8.add(f.readLine());
		}
		f = new BufferedReader(new FileReader("src/data/train_vie.txt"));
		while(f.ready()) {
			g9.add(f.readLine());
		}
		f = new BufferedReader(new FileReader("src/data/train_swe.txt"));
		while(f.ready()) {
			g10.add(f.readLine());
		}
		f = new BufferedReader(new FileReader("src/data/train_arb.txt"));
		while(f.ready()) {
			g11.add(f.readLine());
		}
		f = new BufferedReader(new FileReader("src/data/train_hin.txt"));
		while(f.ready()) {
			g12.add(f.readLine());
		}
		f = new BufferedReader(new FileReader("src/data/train_bng.txt"));
		while(f.ready()) {
			g13.add(f.readLine());
		}
		f = new BufferedReader(new FileReader("src/data/train_ita.txt"));
		while(f.ready()) {
			g14.add(f.readLine());
		}
		run();
	}
	
	/*
	 * print accuracy on test data for selected language
	 */
	public static void trial(int goal) throws IOException {
		@SuppressWarnings("resource")
		BufferedReader f = null;
		switch(goal) {
		case 0: f = new BufferedReader(new FileReader("src/data/test_chi.txt")); break;
		case 1: f = new BufferedReader(new FileReader("src/data/test_jap.txt")); break;
		case 2: f = new BufferedReader(new FileReader("src/data/test_eng.txt")); break;
		case 3: f = new BufferedReader(new FileReader("src/data/test_spn.txt")); break;
		case 4: f = new BufferedReader(new FileReader("src/data/test_kor.txt")); break;
		case 5: f = new BufferedReader(new FileReader("src/data/test_fre.txt")); break;
		case 6: f = new BufferedReader(new FileReader("src/data/test_ger.txt")); break;
		case 7: f = new BufferedReader(new FileReader("src/data/test_rus.txt")); break;
		case 8: f = new BufferedReader(new FileReader("src/data/test_vie.txt")); break;
		case 9: f = new BufferedReader(new FileReader("src/data/test_swe.txt")); break;
		case 10: f = new BufferedReader(new FileReader("src/data/test_arb.txt")); break;
		case 11: f = new BufferedReader(new FileReader("src/data/test_hin.txt")); break;
		case 12: f = new BufferedReader(new FileReader("src/data/test_bng.txt")); break;
		case 13: f = new BufferedReader(new FileReader("src/data/test_ita.txt")); break;
		default:break;
		}
		double ct = 0, ttl = 0;
		ArrayList<String> wrong = new ArrayList<String>();
		ArrayList<String> wrongV = new ArrayList<String>();
		while(f.ready()) {
			String s = f.readLine().toLowerCase();
			test(s);
			int idx = 0;
			for(int i = 1; i < out.length; i++) {if(out[i] > out[idx]) {idx = i;}}
			if(idx == goal) {
				ct++;
			}else {
				wrong.add(s + ",");
				if(idx == 0) {wrongV.add("(MANDARIN)");}
				else if(idx == 1) {wrongV.add("(JAPANESE)");}
				else if(idx == 2) {wrongV.add("(ENGLISH)");}
				else if(idx == 3) {wrongV.add("(SPANISH)");}
				else if(idx == 4) {wrongV.add("(KOREAN)");}
				else if(idx == 5) {wrongV.add("(FRENCH)");}
				else if(idx == 6) {wrongV.add("(GERMAN)");}
				else if(idx == 7) {wrongV.add("(RUSSIAN)");}
				else if(idx == 8) {wrongV.add("(VIETNAMESE)");}
				else if(idx == 9) {wrongV.add("(SWEDISH)");}
				else if(idx == 10) {wrongV.add("(ARABIC)");}
				else if(idx == 11) {wrongV.add("(HINDI)");}
				else if(idx == 12) {wrongV.add("(BENGALI)");}
				else if(idx == 13) {wrongV.add("(ITALIAN)");}
			}
			ttl++;
		}
		System.out.println("MISSED");
		if(wrong.size() == 0) {System.out.println("*none*");}
		else {
			for(int j = 0; j < wrong.size(); j++) {
				System.out.print(wrong.get(j).substring(0,wrong.get(j).indexOf(',')));
				System.out.println(wrongV.get(j));
			}
		}

		System.out.println(ct + " of " + ttl + " CORRECT (" + ct / ttl + ")");
	}
	
	/*
	 * print accuracy on training data for selected language
	 */
	public static void ptrial(int goal) throws IOException {
		@SuppressWarnings("resource")
		BufferedReader f = null;
		switch(goal) {
		case 0: f = new BufferedReader(new FileReader("src/data/train_chi.txt")); break;
		case 1: f = new BufferedReader(new FileReader("src/data/train_jap.txt")); break;
		case 2: f = new BufferedReader(new FileReader("src/data/train_eng.txt")); break;
		case 3: f = new BufferedReader(new FileReader("src/data/train_spn.txt")); break;
		case 4: f = new BufferedReader(new FileReader("src/data/train_kor.txt")); break;
		case 5: f = new BufferedReader(new FileReader("src/data/train_fre.txt")); break;
		case 6: f = new BufferedReader(new FileReader("src/data/train_ger.txt")); break;
		case 7: f = new BufferedReader(new FileReader("src/data/train_rus.txt")); break;
		case 8: f = new BufferedReader(new FileReader("src/data/train_vie.txt")); break;
		case 9: f = new BufferedReader(new FileReader("src/data/train_swe.txt")); break;
		case 10: f = new BufferedReader(new FileReader("src/data/train_arb.txt")); break;
		case 11: f = new BufferedReader(new FileReader("src/data/train_hin.txt")); break;
		case 12: f = new BufferedReader(new FileReader("src/data/train_bng.txt")); break;
		case 13: f = new BufferedReader(new FileReader("src/data/train_ita.txt")); break;
		}
		double ct = 0, ttl = 0;
		while(f.ready()) {
			String s = f.readLine().toLowerCase();
			test(s);
			int idx = 0;
			for(int i = 1; i < out.length; i++) {if(out[i] > out[idx]) {idx = i;}}
			if(idx == goal) {
				ct++;
			}
			ttl++;
		}

		System.out.println(ct + " of " + ttl + " CORRECT (" + ct / ttl + ")");
	}
	
	/*
	 * print accuracy on training data for all testing languages
	 */
	public static void allptrial() throws IOException{

		double ct = 0, ttl = 0;
		for(int j : testing) {
				@SuppressWarnings("resource")
				BufferedReader f = null;
				switch(j) {
				case 0: f = new BufferedReader(new FileReader("src/data/train_chi.txt")); break;
				case 1: f = new BufferedReader(new FileReader("src/data/train_jap.txt")); break;
				case 2: f = new BufferedReader(new FileReader("src/data/train_eng.txt")); break;
				case 3: f = new BufferedReader(new FileReader("src/data/train_spn.txt")); break;
				case 4: f = new BufferedReader(new FileReader("src/data/train_kor.txt")); break;
				case 5: f = new BufferedReader(new FileReader("src/data/train_fre.txt")); break;
				case 6: f = new BufferedReader(new FileReader("src/data/train_ger.txt")); break;
				case 7: f = new BufferedReader(new FileReader("src/data/train_rus.txt")); break;
				case 8: f = new BufferedReader(new FileReader("src/data/train_vie.txt")); break;
				case 9: f = new BufferedReader(new FileReader("src/data/train_swe.txt")); break;
				case 10: f = new BufferedReader(new FileReader("src/data/train_arb.txt")); break;
				case 11: f = new BufferedReader(new FileReader("src/data/train_hin.txt")); break;
				case 12: f = new BufferedReader(new FileReader("src/data/train_bng.txt")); break;
				case 13: f = new BufferedReader(new FileReader("src/data/train_ita.txt")); break;
				}
				while(f.ready()) {
					String s = f.readLine().toLowerCase();
					test(s);
					int idx = 0;
					for(int i = 1; i < out.length; i++) {if(out[i] > out[idx]) {idx = i;}}
					if(idx == j) {
						ct++;
					}
					ttl++;
				}

				
			}
		System.out.println(ct + " of " + ttl + " CORRECT (" + ct / ttl + ")");
	}
	
	/*
	 * print accuracy of test data for all current training languages
	 */
	public static void alltrial() throws IOException {

		double ct = 0, ttl = 0;
		for(int j : testing) {
				@SuppressWarnings("resource")
				BufferedReader f = null;
				switch(j) {
				case 0: f = new BufferedReader(new FileReader("src/data/test_chi.txt")); break;
				case 1: f = new BufferedReader(new FileReader("src/data/test_jap.txt")); break;
				case 2: f = new BufferedReader(new FileReader("src/data/test_eng.txt")); break;
				case 3: f = new BufferedReader(new FileReader("src/data/test_spn.txt")); break;
				case 4: f = new BufferedReader(new FileReader("src/data/test_kor.txt")); break;
				case 5: f = new BufferedReader(new FileReader("src/data/test_fre.txt")); break;
				case 6: f = new BufferedReader(new FileReader("src/data/test_ger.txt")); break;
				case 7: f = new BufferedReader(new FileReader("src/data/test_rus.txt")); break;
				case 8: f = new BufferedReader(new FileReader("src/data/test_vie.txt")); break;
				case 9: f = new BufferedReader(new FileReader("src/data/test_swe.txt")); break;
				case 10: f = new BufferedReader(new FileReader("src/data/test_arb.txt")); break;
				case 11: f = new BufferedReader(new FileReader("src/data/test_hin.txt")); break;
				case 12: f = new BufferedReader(new FileReader("src/data/test_bng.txt")); break;
				case 13: f = new BufferedReader(new FileReader("src/data/test_ita.txt")); break;
				}
				while(f.ready()) {
					String s = f.readLine().toLowerCase();
					test(s);
					int idx = 0;
					for(int i = 1; i < out.length; i++) {if(out[i] > out[idx]) {idx = i;}}
					if(idx == j) {
						ct++;
					}
					ttl++;
				}

				
			}
		System.out.println(ct + " of " + ttl + " CORRECT (" + ct / ttl + ")");
	}
	
	/*
	 * main loop: takes inputs and runs program
	 */
	public static void run() throws IOException {
		Scanner sc = new Scanner(System.in);
		System.out.println("input command:");
		String s = sc.next();
		if(s.equals("train")) {
			System.out.println("how many?");
			int i = Integer.parseInt(sc.next());
			train(i);
			run();
		}else if(s.equals("test")) {
			System.out.println("input test:");
			test(sc.next() + ",");
			int idx = 0;
			for(int i = 1; i < out.length; i++) {if(out[i] > out[idx]) {idx = i;}}
			System.out.print("Prediction:" );
			switch(idx) {
			case 0: System.out.println("MANDARIN"); break;
			case 1: System.out.println("JAPANESE"); break;
			case 2: System.out.println("ENGLISH"); break;
			case 3: System.out.println("SPANISH"); break;
			case 4: System.out.println("KOREAN"); break;
			case 5: System.out.println("FRENCH"); break;
			case 6: System.out.println("GERMAN"); break;
			case 7: System.out.println("RUSSIAN"); break;
			case 8: System.out.println("VIETNAMESE"); break;
			case 9: System.out.println("SWEDISH"); break;
			case 10: System.out.println("ARABIC"); break;
			case 11: System.out.println("HINDI"); break;
			case 12: System.out.println("BENGALI"); break;
			case 13: System.out.println("ITALIAN"); break;
			}
			System.out.print("[" + r(out[0]));
			for(int i = 1; i < out.length; i++) {System.out.print("," + r(out[i]));}
			System.out.println("]");
			run();
		}else if(s.equals("val")) {
			System.out.print("[");
			for(int i = 0; i < in2.length - 1; i++) {
				System.out.print(in2[i] + ",");
			}
			System.out.println(in2[in2.length - 1] + "]");
			run();
		}else if(s.equals("w1")) {
			for(int i = 0; i < weight1.length; i++) {
				for(int j = 0; j < weight1[0].length; j++) {
					for(int k = 0; k < weight1[0][0].length; k++) {
						System.out.print(weight1[i][j][k] + ",");
					}
					System.out.println((char)(j + 'a') + "," + i);
					//System.out.println();
				}
			}
			run();
		}else if(s.equals("w2")) {
			for(int i = 0; i < weight2.length; i++) {
				for(int j = 0; j < weight2[0].length; j++) {
					System.out.print(weight2[i][j] + ",");
				}
				System.out.println();
			}
			run();
		}else if(s.equals("togdetail")) {
			speed = !speed;
			run();
		}else if(s.equals("togacc")) {
			acc = !acc;
			run();
		}else if(s.equals("togcount")) {
			count = !count;
			run();
		}else if(s.equals("trial")) {
			System.out.println("Which language?");
			String st = sc.next();
			if(st.toLowerCase().equals("chi") || st.toLowerCase().equals("chinese") || st.toLowerCase().equals("mandarin") || st.equals("0") ) {
				trial(0);
			}else if(st.toLowerCase().equals("jap") || st.toLowerCase().equals("japanese") || st.equals("1") ) {
				trial(1);
			}else if(st.toLowerCase().equals("eng") || st.toLowerCase().equals("english") || st.equals("2") ) {
				trial(2);
			}else if(st.toLowerCase().equals("spn") || st.toLowerCase().equals("spanish") || st.equals("3") ) {
				trial(3);
			}else if(st.toLowerCase().equals("kor") || st.toLowerCase().equals("korean") || st.equals("4") ) {
				trial(4);
			}else if(st.toLowerCase().equals("fre") || st.toLowerCase().equals("french") || st.equals("5") ) {
				trial(5);
			}else if(st.toLowerCase().equals("ger") || st.toLowerCase().equals("german") || st.equals("6") ) {
				trial(6);
			}else if(st.toLowerCase().equals("rus") || st.toLowerCase().equals("russian") || st.equals("7") ) {
				trial(7);
			}else if(st.toLowerCase().equals("vie") || st.toLowerCase().equals("vietnamese") || st.equals("8") ) {
				trial(8);
			}else if(st.toLowerCase().equals("swe") || st.toLowerCase().equals("swedish") || st.equals("9") ) {
				trial(9);
			}else if(st.toLowerCase().equals("arb") || st.toLowerCase().equals("arabic") || st.equals("10") ) {
				trial(10);
			}else if(st.toLowerCase().equals("hin") || st.toLowerCase().equals("hindi") || st.equals("11") ) {
				trial(11);
			}else if(st.toLowerCase().equals("bng") || st.toLowerCase().equals("bengali") || st.equals("12") ) {
				trial(12);
			}else if(st.toLowerCase().equals("ita") || st.toLowerCase().equals("italian") || st.equals("13") ) {
				trial(13);
			}
			run();
		}else if(s.equals("ptrial")) {
			System.out.println("Which language?");
			String st = sc.next();
			if(st.toLowerCase().equals("chi") || st.toLowerCase().equals("chinese") || st.toLowerCase().equals("mandarin") || st.equals("0") ) {
				ptrial(0);
			}else if(st.toLowerCase().equals("jap") || st.toLowerCase().equals("japanese") || st.equals("1") ) {
				ptrial(1);
			}else if(st.toLowerCase().equals("eng") || st.toLowerCase().equals("english") || st.equals("2") ) {
				ptrial(2);
			}else if(st.toLowerCase().equals("spn") || st.toLowerCase().equals("spanish") || st.equals("3") ) {
				ptrial(3);
			}else if(st.toLowerCase().equals("kor") || st.toLowerCase().equals("korean") || st.equals("4") ) {
				ptrial(4);
			}else if(st.toLowerCase().equals("fre") || st.toLowerCase().equals("french") || st.equals("5") ) {
				ptrial(5);
			}else if(st.toLowerCase().equals("ger") || st.toLowerCase().equals("german") || st.equals("6") ) {
				ptrial(6);
			}else if(st.toLowerCase().equals("rus") || st.toLowerCase().equals("russian") || st.equals("7") ) {
				ptrial(7);
			}else if(st.toLowerCase().equals("vie") || st.toLowerCase().equals("vietnamese") || st.equals("8") ) {
				ptrial(8);
			}else if(st.toLowerCase().equals("swe") || st.toLowerCase().equals("swedish") || st.equals("9") ) {
				ptrial(9);
			}else if(st.toLowerCase().equals("arb") || st.toLowerCase().equals("arabic") || st.equals("10") ) {
				ptrial(10);
			}else if(st.toLowerCase().equals("hin") || st.toLowerCase().equals("hindi") || st.equals("11") ) {
				ptrial(11);
			}else if(st.toLowerCase().equals("bng") || st.toLowerCase().equals("bengali") || st.equals("12") ) {
				ptrial(12);
			}else if(st.toLowerCase().equals("ita") || st.toLowerCase().equals("italian") || st.equals("13") ) {
				ptrial(13);
			}
			run();
		}else if(s.equals("alltrial")) {
			alltrial();
			run();
		}else if(s.equals("allptrial")) {
			allptrial();
			run();
		}else if(s.equals("ptrain")) {
			System.out.println("Which language?");
			String st = sc.next();
			int[] temp = testing;
			if(st.toLowerCase().equals("chi") || st.toLowerCase().equals("chinese") || st.toLowerCase().equals("mandarin") || st.equals("0") ) {
				testing = new int[] {0};
			}else if(st.toLowerCase().equals("jap") || st.toLowerCase().equals("japanese") || st.equals("1") ) {
				testing = new int[] {1};
			}else if(st.toLowerCase().equals("eng") || st.toLowerCase().equals("english") || st.equals("2") ) {
				testing = new int[] {2};
			}else if(st.toLowerCase().equals("spn") || st.toLowerCase().equals("spanish") || st.equals("3") ) {
				testing = new int[] {3};
			}else if(st.toLowerCase().equals("kor") || st.toLowerCase().equals("korean") || st.equals("4") ) {
				testing = new int[] {4};
			}else if(st.toLowerCase().equals("fre") || st.toLowerCase().equals("french") || st.equals("5") ) {
				testing = new int[] {5};
			}else if(st.toLowerCase().equals("ger") || st.toLowerCase().equals("german") || st.equals("6") ) {
				testing = new int[] {6};
			}else if(st.toLowerCase().equals("rus") || st.toLowerCase().equals("russian") || st.equals("7") ) {
				testing = new int[] {7};
			}else if(st.toLowerCase().equals("vie") || st.toLowerCase().equals("vietnamese") || st.equals("8") ) {
				testing = new int[] {8};
			}else if(st.toLowerCase().equals("swe") || st.toLowerCase().equals("swedish") || st.equals("9") ) {
				testing = new int[] {9};
			}else if(st.toLowerCase().equals("arb") || st.toLowerCase().equals("arabic") || st.equals("10") ) {
				testing = new int[] {10};
			}else if(st.toLowerCase().equals("hin") || st.toLowerCase().equals("hindi") || st.equals("11") ) {
				testing = new int[] {11};
			}else if(st.toLowerCase().equals("bng") || st.toLowerCase().equals("bengali") || st.equals("12") ) {
				testing = new int[] {12};
			}else if(st.toLowerCase().equals("ita") || st.toLowerCase().equals("italian") || st.equals("13") ) {
				testing = new int[] {13};
			}
			System.out.println("how many?");
			int i = Integer.parseInt(sc.next());
			train(i);
			testing = temp;
			run();
		}else if(s.equals("ptrainm")) {
			System.out.println("Which language?");
			String st = sc.next();
			int[] temp = testing;
			if(st.toLowerCase().equals("chi") || st.toLowerCase().equals("chinese") || st.toLowerCase().equals("mandarin") || st.equals("0") ) {
				testing = new int[] {0};
			}else if(st.toLowerCase().equals("jap") || st.toLowerCase().equals("japanese") || st.equals("1") ) {
				testing = new int[] {1};
			}else if(st.toLowerCase().equals("eng") || st.toLowerCase().equals("english") || st.equals("2") ) {
				testing = new int[] {2};
			}else if(st.toLowerCase().equals("spn") || st.toLowerCase().equals("spanish") || st.equals("3") ) {
				testing = new int[] {3};
			}else if(st.toLowerCase().equals("kor") || st.toLowerCase().equals("korean") || st.equals("4") ) {
				testing = new int[] {4};
			}else if(st.toLowerCase().equals("fre") || st.toLowerCase().equals("french") || st.equals("5") ) {
				testing = new int[] {5};
			}else if(st.toLowerCase().equals("ger") || st.toLowerCase().equals("german") || st.equals("6") ) {
				testing = new int[] {6};
			}else if(st.toLowerCase().equals("rus") || st.toLowerCase().equals("russian") || st.equals("7") ) {
				testing = new int[] {7};
			}else if(st.toLowerCase().equals("vie") || st.toLowerCase().equals("vietnamese") || st.equals("8") ) {
				testing = new int[] {8};
			}else if(st.toLowerCase().equals("swe") || st.toLowerCase().equals("swedish") || st.equals("9") ) {
				testing = new int[] {9};
			}else if(st.toLowerCase().equals("arb") || st.toLowerCase().equals("arabic") || st.equals("10") ) {
				testing = new int[] {10};
			}else if(st.toLowerCase().equals("hin") || st.toLowerCase().equals("hindi") || st.equals("11") ) {
				testing = new int[] {11};
			}else if(st.toLowerCase().equals("bng") || st.toLowerCase().equals("bengali") || st.equals("12") ) {
				testing = new int[] {12};
			}else if(st.toLowerCase().equals("ita") || st.toLowerCase().equals("italian") || st.equals("13") ) {
				testing = new int[] {13};
			}
			a = 0;
			double tempstep = step;
			step = .85;
			while(a < .95) {
				train(1);
			}
			step = tempstep;
			testing = temp;
			run();
		}else if(s.equals("togstep")) {
			System.out.println("s value?");
			step = Double.parseDouble(sc.next());
			run();
		}else if(!s.equals("exit")) {
			System.out.println("unknown command!");
			run();
		}
		sc.close();
	}
}
