#include "main.h"
/**
 * shell_loop_hsh - loop and accept command input
 * @argc: 2D array to stor input command
 * @argv: argument vector
 * @mode: open mode
 * Return: on exit 0, error -1, success 1
 */
int shell_loop_hsh( )
{
char *buffer;
char **av_token;
int result, w_len;
ssize_t read;
size_t len;
do {
len = w_len = 0;
av_token = NULL;
buffer = NULL;
if (isatty(STDIN_FILENO))
_putchar('$');
av_token = create_2D_buffer(av_token);
buffer = (char *)malloc(sizeof(char) * BUFFER_SIZE);
read = getline(&buffer, &len, stdin);
if (feof(stdin) || (read <= 1))
{
result = -1;
_free_all(buffer, av_token, MAX_WORDS);
continue;
}
buffer[_strcspn(buffer, '\n')] = '\0';
result = strcmp(buffer,  "exit");
if (result == 0)
{
_free_all(buffer, av_token, MAX_WORDS);
_exit_shell(environ);
}
if (strcmp(buffer, "printenv") == 0)
_printenv(environ);
if (read > 0)
{
av_token = tokenize_string(buffer, av_token, &w_len);
_execve(av_token, NULL);
}
_free_all(buffer, av_token, MAX_WORDS);
} while (result != 0 && read != -1 && buffer[read - 1] != '\n');
return (-1);
}
