#include "wav_header.h"  /* Include the header (not strictly necessary here) */
#include <string.h>
#include <stdio.h>
#include <sndfile.h>
#include <stdlib.h>


int wav_read(char wav_dir_path[], char wav_file_name[]){    /* Function definition */
	
	char wav_file_full_path[100];
	sprintf(wav_file_full_path, "%s%s%s", wav_dir_path, "/", wav_file_name);
	printf("\nReading wav file from path %s\n", wav_file_full_path);
	

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
//    sf = sf_open("/home/mihai/Munca/ArhivaMunca/TimitClean1SecV2/FAEM0_16kHz.wav",SFM_READ,&info);
	sf = sf_open(wav_file_full_path, SFM_READ, &info);
	if (sf == NULL)
        {
        printf("Failed to open the file.\n");
        exit(-1);
        }

    /* Print some of the info, and figure out how much data to read. */

	f = info.frames;
    sr = info.samplerate;
    c = info.channels;
    printf("f  = frames     = %d\n",f);
    printf("sr = samplerate = %d\n",sr); // sr = Sample Rate -> variabila care stocheaza valoarea frecventei de esantionare
    printf("c  = channels   = %d\n",c);
    num_items = f*c;
    printf("num_items = f*c = %d\n",num_items);
	
    /* Allocate space for the data to be read, then read it. */
    
	buf = (int *) malloc(num_items*sizeof(int)); // buf -> bufferul care contine informatia utila
    num = sf_read_int(sf,buf,num_items);
    sf_close(sf);
    printf("Read %d items\n",num);
  
    /* Write the data to filedata.out. */

	out = fopen("filedata.out","w");
    for (i = 0; i < num; i += c)
        {
        for (j = 0; j < c; ++j)
            fprintf(out,"%d ",buf[i+j]/65536);
        fprintf(out,"\n");
        }
    fclose(out);
	
//	for(int i = 0; i < 200; i++){
//		printf("  buf[%d] = %d  ", i, buf[i]/65536);
//	}
	printf("\n\nUltima val din vector %d\n", buf[667829]/65536);
	
	printf("\n buf = %p\n", buf);
	
	return 0;
	
	
}



void wav_read2(char wav_dir_path[], char wav_file_name[], int **result_array, int **length, int **samp_frec){    /* Function definition */
	
	char wav_file_full_path[100];
	sprintf(wav_file_full_path, "%s%s%s", wav_dir_path, "/", wav_file_name);
	printf("\nReading wav file from path %s\n", wav_file_full_path);
	

//	data_file = malloc(sizeof(wav));
	
	
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
//    sf = sf_open("/home/mihai/Munca/ArhivaMunca/TimitClean1SecV2/FAEM0_16kHz.wav",SFM_READ,&info);
	sf = sf_open(wav_file_full_path, SFM_READ, &info);
	if (sf == NULL)
        {
        printf("Failed to open the file.\n");
        exit(-1);
        }

    /* Print some of the info, and figure out how much data to read. */

	f = info.frames;
    *samp_frec = info.samplerate; //samp_frec = &sr;
    c = info.channels;
    printf("f  = frames     = %d\n",f);
    printf("sr = samplerate = %d\n",sr); // sr = Sample Rate -> variabila care stocheaza valoarea frecventei de esantionare
    printf("c  = channels   = %d\n",c);
    num_items = f*c;
	*length = f*c;
    printf("num_items = f*c = %d\n", num_items);
	
    /* Allocate space for the data to be read, then read it. */
    
//	buf = (int *) malloc(num_items*sizeof(int)); // buf -> bufferul care contine informatia utila
	*result_array = (int *) malloc(num_items*sizeof(int));
	
    num = sf_read_int(sf, *result_array, num_items);
	printf("\nnum = %d\n", num);
	
    sf_close(sf);
    printf("Read %d items\n", num);
	
    /* Write the data to filedata.out. */

	
	out = fopen("filedata.out","w");
    for (i = 0; i < num; i += c)
        {
        for (j = 0; j < c; ++j)
            fprintf(out, "%d ", (*result_array)[i+j]/65536);
        fprintf(out, "\n");
        }
    
	fclose(out);

		// Afisari ulterioare
	for(int i = 0; i < 10; i++){
		printf("\n  result_array[%d] = %d  \n", i, (*result_array)[i]/65536);
	}
/*	
	printf("\n\nUltima val din vector %d\n", array[667829]/65536);
*/
}
	