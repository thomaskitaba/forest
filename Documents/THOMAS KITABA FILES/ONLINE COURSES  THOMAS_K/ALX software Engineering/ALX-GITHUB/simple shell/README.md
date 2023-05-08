#################### folow of the program ####################
# int hsh(int argc, char **argv);                            #
# int shell_loop_hsh(void);                                  #
# char **tokenize_string(char *str, char **token, int *len); #
# int _strtok(char *str, char *delim, char **token);         #
# char *add_path(char *path_arg, char *path_buffer);         #
# char *add_path(char *path_arg, char *path_buffer);         #
# int _execve(char **av, char **env);                        #
##############################################################

##########MAIN SHELL PART############
int _execve(char **av, char **env);
-> Input: 
-> Output:
int shell_loop_hsh(void);
-> Input: 
-> Output:
int _strtok(char *str, char *delim, char **token);
-> Input: 
-> Output:

########## String Manuplation ############
int _strlen(char *str);
-> Input: 
-> Output:
char *_strcat(char *dest, char *src);
-> Input: 
-> Output:
int _strcmp(char *s1, char *s2);
-> Input: 
-> Output:
int _strcspn(char *buf, char c);
-> Input: 
-> Output:
int _get_word_count(char *str, char *delim);
-> Input: 
-> Output:

########## memory managment ############

void _free_2D(char **token, int rows);
-> Input: 
-> Output:
void _free_all(char *token, char **av_token, int w_len);
-> Input: 
-> Output:
char *create_buffer(void);
-> Input: 
-> Output:
char **create_2D_buffer(char **av);
-> Input: 
-> Output:

##########MAIN SHELL PART############

char **tokenize_string(char *str, char **token, int *len);
-> Input: 
-> Output:
int token(char **av, char *string);
-> Input: 
-> Output:
int _tokenize(char **av, char **argv, int argc);
-> Input: 
-> Output:
#########  Tokenizer and helper ###########

char *add_path(char *path_arg, char *path_buffer);
-> Input: 
-> Output:
char *path_helper(char *p_a, char *p_b, int n_c, int d_len, int p_len);
-> Input: 
-> Output:

##########built ins ############
int _exit_shell(char **string);
-> Input: 
-> Output:
int (*get_builtin_cmd(char *s))(char **);
-> Input: 
-> Output:
int _printenv(char **env);
-> Input: 
-> Output:

########## Supporting and testig ############
void _print_2D(char **av, int argc);
-> Input: 
-> Output: