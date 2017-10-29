#include "VADlib.h"  /* Include the header (not strictly necessary here) */
#include <string.h>
#include <stdio.h>
#include <sndfile.h>
#include <stdlib.h>
#include <math.h>

struct element{
	double data; // campul denumit nr reprezinta numarul de pe bila extrasa
	struct element *next; // campul "next" reprezinta o adresa catre urmatorul nod (bila) din lista
};

struct Vector{
	unsigned int lungime;
	struct element *header, *curent;
};


void make_vector(struct Vector *vector){ // functie de creare a unei liste: se aloca memorie pentru nodul de referinta -> cap

	vector->header = (struct element*)malloc(sizeof(struct element));
	
	if(vector->header==NULL){
		printf("\nVector initialisation: Failed due to insufficient memory\n"); 
	}else{
		vector->header->next = NULL; 
		vector->header->data = 0;
		vector->lungime = 0;
		vector->curent = vector->header;
		printf("\nVector initialisation: Success\n");
	}
}

void add_elem(struct Vector *vector, double data){
	
	struct element *new_elem;
	new_elem = (struct element*)malloc(sizeof(struct element));
	
	if (new_elem == NULL){ printf("\n memorie insuficienta\n"); }
	else{
		vector->curent = vector->header;
		while (vector->curent->next != NULL){
			vector->curent = vector->curent->next;
		}
		new_elem->next = vector->curent->next;
		vector->curent->next = new_elem;
		
		new_elem->data = data;
		vector->lungime++;
		
		vector->curent = new_elem;
	}
}

void show_vector(struct Vector *vector){
	
	if(vector->header->next == NULL){ printf("\nVectorul este gol\n");}
	
	else{
		
		vector->curent = vector->header->next;
		
		int i = 0;
		while(vector->curent != NULL){
			printf("\ndata[%d] = %f\n", i, vector->curent->data);
			i++;
			vector->curent = vector->curent->next;
		}
	}
}

double get_index_value(struct Vector *vector, unsigned long index){
	
	if(vector->header->next == NULL){ printf("\nVectorul este gol\n"); return 0;}
	else{
		unsigned long i = 0;
		
		vector->curent = vector->header->next;
		while(i < index){
			if(vector->curent->next == NULL && i != index-1){
				printf("\nAi ajuns la finalul vectorului!\n");
				break;
			}
			else{
				vector->curent = vector->curent->next;
				i++;
			}
		}
	}
	return vector->curent->data;
}

int sign(int x) {
    return (x > 0) - (x < 0);
}

struct Vector *short_term_power(int *sgn, unsigned int sgn_len, unsigned int samp_frec, double frame_time, double stop_time, double alpha){
	
	printf("\n-------------- BEGIN OF Short Term Power ALGORITHM --------------\n");
	
	unsigned long samp_per_frame = ((double)samp_frec) * frame_time;
	unsigned long frames_nr = floor(sgn_len / samp_per_frame);
	unsigned int voice_nr_of_elem = samp_per_frame * frames_nr;	
	int square_sgn_value[voice_nr_of_elem];	
	unsigned long stop_frame = stop_time / frame_time; 
	double threshold_elem_sum = 0;
	
	printf("\nSampPerFrame = %lu ", samp_per_frame);
	printf("\nFramesNr = %lu ", frames_nr);	
	printf("\nThershold frames nr evaluation = %lu\n", stop_frame);	
	
	struct Vector *power = (struct Vector*)malloc(sizeof(struct Vector));
	make_vector(power);

//	FILE *power_file = fopen("STP_powerEvo.txt","w"); // Creare/Deschidere (cu suprascriere) a fisierului cu evolutia puterii semnalului pe ferestre
	
	for (int i = 0; i < frames_nr; i++){		// ALGORITMUL DE CALCUL AL PUTERII SEMNALULUI PE FERESTRE

		unsigned long start = i*samp_per_frame; unsigned long stop = (i+1)*samp_per_frame; // Calcul indecsi ferestre
	    double sum = 0;
		
	    for (int j = start; j < stop; j++){
			square_sgn_value[j] = sgn[j]/65536;
	    	sum += square_sgn_value[j]*square_sgn_value[j];
	    }
		
		if (i > 0){
			sum = sum * alpha + get_index_value(power, i-1) * (1 - alpha);	// Formula de netezire
		}
			
//		fprintf(power_file, "%f,", sum);
		
	    add_elem(power, sum);
		if (i < stop_frame){
			threshold_elem_sum += sum;
		}
	}
	
//	fclose(power_file);	// Inchiderea fisierului cu evolutia puterii pe ferestre a semnalului
	
	
	double threshold = threshold_elem_sum / stop_frame * 1.5;  // calcul prag cu ajustare -> *1.5
	printf("\nThreshold value: %f\n", threshold);
	
	unsigned short voice_det[voice_nr_of_elem];
	
				//	SAVING DATA FILES FOR VOICE/NON-VOICE SIGNAL SAMPLE
//	FILE *bin_file = fopen("/home/mihai/cProjects/wavReading/execs/STP.bin", "wb"); // Creare/deschidere fisier in mod scriere binar
//	FILE *txt_file = fopen("/home/mihai/cProjects/wavReading/execs/STP.txt", "w");	// Creare/deschidere fisier text pentru scrierea deciziei de voce
	
	for (int i = 0; i < frames_nr; i++){		//  REALIZARE VECTOR 1/0 PENTRU VOCE/ABSENTA VOCE

		if (get_index_value(power, i) > threshold){
			for (int j = ((unsigned long)i)*samp_per_frame; j < ((unsigned long)i+1)*samp_per_frame; j++){
				voice_det[j] = 1;
//				fprintf(bin_file, "%d ", voice_det[j]);
//				fprintf(txt_file, "%d ", voice_det[j]);
			}
		}
		else{
			for (int j = i*samp_per_frame; j < (i+1)*samp_per_frame; j++){
				voice_det[j] = 0;
//				fprintf(bin_file, "%d ", voice_det[j]);
//				fprintf(txt_file, "%d ", voice_det[j]);
			}
		}
	}

//	fclose(bin_file);	
//	fclose(txt_file);
	
	printf("\n______________END OF Short Term Power ALGORITHM______________	\n");
	
	return power;
}

struct Vector *zero_crossing_rate(int *sgn, unsigned int sgn_len, unsigned int samp_frec, double frame_time, double stop_time, double alpha){
	
	printf("\n------------- BEGIN OF Zero Crossing Rate ALGORITHM--------------\n");
	
	unsigned long samp_per_frame = ((double)samp_frec) * frame_time;
	unsigned long frames_nr = floor(sgn_len / samp_per_frame);
	unsigned int voice_nr_of_elem = samp_per_frame * frames_nr;		
	unsigned long stop_frame = stop_time / frame_time;	
	double threshold_elem_sum = 0;
	
	printf("\nSampPerFrame = %lu ", samp_per_frame);
	printf("\nFramesNr = %lu ", frames_nr);
	printf("\nThershold frames nr evaluation = %lu\n", stop_frame);	
	
	struct Vector *zcr = (struct Vector*)malloc(sizeof(struct Vector));
	make_vector(zcr);
	
//	FILE *zcr_file = fopen("ZCR_zcrEvo.txt", "w");
	
	for (int i = 0; i < frames_nr; i++){
		
		unsigned long start = i*samp_per_frame; unsigned long stop = (i+1)*samp_per_frame; // Calcul indecsi ferestre	
		double sum = 0;
		
		for (int j = start; j < stop-1; j++){	
			if ( ( sign(sgn[j]) != sign(sgn[j+1]) ) && (sign(sgn[j]) != 0)){
				sum += 1;
			}
		}
	    
		if (i > 0){
			sum = sum * alpha + get_index_value(zcr, i-1) * (1 - alpha);	// Formula de netezire
		}
		
//		fprintf(zcr_file, "%f,", sum);
		add_elem(zcr, sum);
		
		if (i < stop_frame){
			threshold_elem_sum += sum;
		}
		
	}
	
//	fclose(zcr_file);
	
	double threshold = threshold_elem_sum / stop_frame * 0.9;  // calcul prag cu ajustare -> *1.5
	printf("\nThreshold value: %f\n", threshold);
	
	unsigned short voice_det[voice_nr_of_elem];
	
				//	SAVING DATA FILES FOR VOICE/NON-VOICE SIGNAL SAMPLE
//	FILE *bin_file = fopen("/home/mihai/cProjects/wavReading/execs/ZCR.bin", "wb"); // Creare/deschidere fisier in mod scriere binar
//	FILE *txt_file = fopen("/home/mihai/cProjects/wavReading/execs/ZCR.txt", "w");	// Creare/deschidere fisier text pentru scrierea deciziei de voce
	
	for (int i = 0; i < frames_nr; i++){		//  REALIZARE VECTOR 1/0 PENTRU VOCE/ABSENTA VOCE

		if (get_index_value(zcr, i) < threshold){
			for (int j = ((unsigned long)i)*samp_per_frame; j < ((unsigned long)i+1)*samp_per_frame; j++){
				voice_det[j] = 1;
//				fprintf(bin_file, "%d ", voice_det[j]);
//				fprintf(txt_file, "%d ", voice_det[j]);
			}
		}
		else{
			for (int j = i*samp_per_frame; j < (i+1)*samp_per_frame; j++){
				voice_det[j] = 0;
//				fprintf(bin_file, "%d ", voice_det[j]);
//				fprintf(txt_file, "%d ", voice_det[j]);
			}
		}
	}	

//	fclose(bin_file);	
//	fclose(txt_file);
	
	printf("\n______________END OF Zero Crossing Rate ALGORITHM______________\n");

	return zcr;
}

void zrmse(int *sgn, unsigned int sgn_len, unsigned int samp_frec, double frame_time, double stop_time){
	
	printf("\n-------------- BEGIN OF Zrmse Algorithm --------------");
	
	const double alpha = 0.5;
	
	unsigned long samp_per_frame = ((double)samp_frec) * frame_time;	printf("\nSampPerFrame = %lu ", samp_per_frame);
	unsigned long frames_nr = floor(sgn_len / samp_per_frame);			printf("\nFramesNr = %lu ", frames_nr);
	unsigned int voice_nr_of_elem = samp_per_frame * frames_nr;
	unsigned long stop_frame = stop_time / frame_time; printf("\nThershold frames nr evaluation = %lu\n", stop_frame);	

	struct Vector *stp, *zcr;
	
	stp = short_term_power(sgn, sgn_len, samp_frec, frame_time, stop_time, alpha);
	zcr = zero_crossing_rate(sgn, sgn_len, samp_frec, frame_time, stop_time, alpha);
	
//	printf("\n stp = %f \n", get_index_value(stp, 1));
//	printf("\n zcr = %f \n", get_index_value(zcr, 1));
	
	struct Vector *zrmse = (struct Vector*)malloc(sizeof(struct Vector));
	make_vector(zrmse);
	
//	FILE *zrmse_file = fopen("zrmseEvo.txt", "w");
	
	double sum = 0;
	for (unsigned long i = 0; i < frames_nr; i++){
		double aux = get_index_value(stp, i) / get_index_value(zcr, i);
		add_elem(zrmse, aux);
//		fprintf(zrmse_file, "%f,", aux);
		if (i < stop_frame){
			sum += aux;
		}
	}
	
//	fclose(zrmse_file);
	
	double threshold = sum / stop_frame * 1.5;
	unsigned short voice_det[voice_nr_of_elem];
	
	printf("\nThreshold value: %f\n", threshold);
	
			//	SAVIND DATA FILES FOR VOICE/NON-VOICE SIGNAL SAMPLE
//	FILE *bin_file = fopen("/home/mihai/cProjects/wavReading/execs/ZRMSE.bin", "wb"); // Creare/deschidere fisier in mod scriere binar
//	FILE *txt_file = fopen("/home/mihai/cProjects/wavReading/execs/ZRMSE.txt", "w");	// Creare/deschidere fisier text pentru scrierea deciziei de voce
	
	for (int i = 0; i < frames_nr; i++){		//  REALIZARE VECTOR 1/0 PENTRU VOCE/ABSENTA VOCE

		if (get_index_value(zrmse, i) > threshold){
			for (int j = ((unsigned long)i)*samp_per_frame; j < ((unsigned long)i+1)*samp_per_frame; j++){
				voice_det[j] = 1;
//				fprintf(bin_file, "%d ", voice_det[j]);
//				fprintf(txt_file, "%d ", voice_det[j]);

			}
		}
		else{
			for (int j = i*samp_per_frame; j < (i+1)*samp_per_frame; j++){
				voice_det[j] = 0;
//				fprintf(bin_file, "%d ", voice_det[j]);
//				fprintf(txt_file, "%d ", voice_det[j]);
			}
		}
	}

//	fclose(bin_file);	
//	fclose(txt_file);
	
	printf("\n______________END OF Zrmse Algorithm______________\n");
}

















