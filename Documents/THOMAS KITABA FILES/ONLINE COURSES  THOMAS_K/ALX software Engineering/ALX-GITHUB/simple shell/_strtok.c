#include "main.h"
/**
 * _strtok - string to words
 * @str: string
 * @delim: delimeter
 * @token: 2D ponter 2 hold tokenized cmd
 * Return: int
 */
int _strtok(char *str, char *delim, char **token)
{
int w_count, total_w_len;
char *word, *str_cpy, *buffer;
if (!str || !delim)
return (-1);
buffer = NULL;
buffer = (char *)malloc(sizeof(char) * BUFFER_SIZE);
w_count = total_w_len = 0;
str_cpy = strdup(str);

word = strtok(str_cpy, delim);
buffer = add_path(word, buffer);
/*if path was corrected and added */
while (word && w_count <= WORD_COUNT)
{
if (w_count == 0 && buffer)
strcpy(token[w_count], buffer);
else
strcpy(token[w_count], word);
w_count++;
word = strtok(NULL, delim);
}
token[w_count] = NULL;
/*TODO: add token to info_t->argc   and info_t->argv*/
free(buffer);
free(str_cpy);
return (w_count);
}
