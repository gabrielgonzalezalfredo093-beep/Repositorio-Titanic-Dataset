## Instalamos librerias necesarias 
install.packages("usethis")
install.packages("gitcreds")

## Cargamos libreria y configuramos
library(usethis)
use_git_config(
  user.name = "anyicastillo", # Cambia esto por tu nombre
  user.email = "anyimaster65@gmail.com" # Usa el email asociado a tu cuenta de GitHub
)

## Creamos token para credenciales de github
create_github_token()

## Configuramos token
library(gitcreds)
gitcreds_set()
