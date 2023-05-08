#include "main.h"
/**
 * tokenize_string - change string to 2D array
 * @str: string to be tokenized
 * @token: token
 * @len: length
 * Return: 1 on success, -1 on failur
 */
char **tokenize_string(char *str, char **token, int *len)
{
/*char str[] = "thomas kitaba feyissa";*/
char *delimeter = " ";
int w_count;
w_count = 0;
/*malloc for 2d array **token  row*/
w_count = _strtok(str, delimeter, token);
if (w_count == -1)
return (NULL);
*len = (w_count + 1);
/*free memory */
return (token);
}
/**
 * token - change word to 2d word
 * @string: string to be converted
 * @av: argument vector
 * Return: 1 on success, -1 on faliur
 */
int token(char **av, char *string)
{
strcpy(av[0], string);
_free_2D(av, WORD_SIZE);
return (1);
}
