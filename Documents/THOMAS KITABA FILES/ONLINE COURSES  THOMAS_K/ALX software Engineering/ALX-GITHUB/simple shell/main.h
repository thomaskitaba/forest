#ifndef MAIN_H
#define MAIN_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <limits.h>
#include <fcntl.h>
#include <errno.h>

#define MAX_WORDS 100
#define MAX_WORD_LENGTH 1024
#define WORD_SIZE 1024
#define WORD_COUNT 100
#define BUFFER_SIZE 1024
extern char **environ;
/**
 * struct builtin - Struct op
 *
 * @string: The string
 * @builtin_f: The function associated
 */
typedef struct builtin
{
char *string;
int (*builtin_f)(char **);
} built_in;

int _putchar(char c);
int _execve(char **av, char **env);
int shell_loop_hsh();
int _strtok(char *str, char *delim, char **token);
int _strlen(char *str);
char *_strcat(char *dest, char *src);
int _strcmp(char *s1, char *s2);
void _free_2D(char **token, int rows);
void _free_all(char *token, char **av_token, int w_len);
char *create_buffer(void);
char **create_2D_buffer(char **av);
int _get_word_count(char *str, char *delim);
int (*get_builtin_cmd(char *s))(char **);
int _strcspn(char *buf, char c);
int hsh(int argc, char **argv);
char **tokenize_string(char *str, char **token, int *len);
int token(char **av, char *string);
int _exit_shell(char **string);
int _tokenize(char **av, char **argv, int argc);
int _strcspn(char *buf, char c);
char *add_path(char *path_arg, char *path_buffer);
void _print_2D(char **av, int argc);
int _printenv(char **env);
char *path_helper(char *p_a, char *p_b, int n_c, int d_len, int p_len);

#endif
