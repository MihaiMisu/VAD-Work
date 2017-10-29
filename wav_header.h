#ifndef WAV_HEADER_H_   /* Include guard */
#define WAV_HEADER_H_

int wav_read(char path[], char file_name[]);  /* An example function declaration */
void wav_read2(char path[], char file_name[], int **wav_pointer_array, int **wav_length, int **samp_frec);

#endif // FOO_H_