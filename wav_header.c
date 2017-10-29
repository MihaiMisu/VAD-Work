#include "wav_header.h"  /* Include the header (not strictly necessary here) */
#include <string.h>
#include <stdio.h>
#include <sndfile.h>
#include <stdlib.h>


void wav_read(char wav_dir_path[], char wav_file_name[], int **result_array, int **length, int **samp_frec){    /* Function definition */
	
	char wav_file_full_path[100];
	sprintf(wav_file_full_path, "%s%s%s", wav_dir_path, "/", wav_file_name);

//	printf("\nReading wav file from path %s\n", wav_file_full_path);
	
	*samp_frec = (int *)malloc(sizeof(int));
	*length = (int *)malloc(sizeof(int));
	if (samp_frec == NULL || length == NULL){
		printf("\n\n Nu s-a alocat memorie pentru sampFrec sau length\n\n");
		exit(-1);
	}

	SNDFILE *sf;
    SF_INFO info;
    int num_channels;
    int num, num_items;
    int *buf;
    int f,sr,c;
    int i,j;
    FILE *out;
	
    /* Open the WAV file. */

	info.format = 0;
	sf = sf_open(wav_file_full_path, SFM_READ, &info);
	if (sf == NULL)
        {
        printf("Failed to open the file.\n");
        exit(-1);
        }

    /* Print some of the info, and figure out how much data to read. */

	f = info.frames;
    *samp_frec = info.samplerate;
    c = info.channels;
//    printf("f  = frames     = %d\n",f);
//    printf("sr = samplerate = %d\n",sr); // sr = Sample Rate -> variabila care stocheaza valoarea frecventei de esantionare
//    printf("c  = channels   = %d\n",c);
    num_items = f*c;
	*length = (num_items);
//    printf("num_items = f*c = %d\n", num_items);
	
    /* Allocate space for the data to be read, then read it. */
    
//	buf = (int *) malloc(num_items*sizeof(int)); // buf -> bufferul care contine informatia utila
	*result_array = (int *) malloc(num_items*sizeof(int));
	
    num = sf_read_int(sf, *result_array, num_items);
//	printf("\nnum = %d\n", num);
	
    sf_close(sf);
//    printf("Read %d items\n", num);
	
    /* Write the data to filedata.out. */
/*
	
	out = fopen("powerEvo.txt","w");
    for (i = 0; i < num; i += c)
        {
        for (j = 0; j < c; ++j)
            fprintf(out, "%d ", (*result_array)[i+j]/65536);
        fprintf(out, "\n");
        }
    
	fclose(out);
	
*/
	
		// Afisari ulterioare
//	for(int i = 0; i < 10; i++){
//		printf("\n  result_array[%d] = %d  \n", i, (*result_array)[i]/65536);
//	}
	

}
	