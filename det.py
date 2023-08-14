import pygame
import random

# Initialize game
pygame.init()
screen = pygame.display.set_mode((400,400))
pygame.display.set_caption('Snake')

# Game variables
snake_pos = [200, 200]
snake_body = [[200, 200], [190, 200], [180, 200]]
food_pos = [random.randrange(1, 40)*10, random.randrange(1, 40)*10]
food_spawn = True
direction = 'RIGHT'
change_direction = direction
score = 0


# Game loop
running = True
while running:

  # Handle events
  for event in pygame.event.get():
    # if event.type == pygame.QUIT:
    #   running = False

    # Change snake direction
    if event.type == pygame.KEYDOWN:
      if event.key == pygame.K_LEFT:
        change_direction = 'LEFT'
      if event.key == pygame.K_RIGHT:
        change_direction = 'RIGHT'
      if event.key == pygame.K_UP:
        change_direction = 'UP'
      if event.key == pygame.K_DOWN:
        change_direction = 'DOWN'

  # Move snake
  if change_direction == 'LEFT' and direction != 'RIGHT':
    direction = 'LEFT'
  if change_direction == 'RIGHT' and direction != 'LEFT':
    direction = 'RIGHT'
  if change_direction == 'UP' and direction != 'DOWN':
    direction = 'UP'
  if change_direction == 'DOWN' and direction != 'UP':
    direction = 'DOWN'

  if direction == 'LEFT':
    snake_pos[0] -= 10
  if direction == 'RIGHT':
    snake_pos[0] += 10
  if direction == 'UP':
    snake_pos[1] -= 10
  if direction == 'DOWN':
    snake_pos[1] += 10

  # Snake body mechanism
  snake_body.insert(0, list(snake_pos))
  if snake_pos[0] == food_pos[0] and snake_pos[1] == food_pos[1]:
    score += 1
    food_spawn = False
  else:
    snake_body.pop()

  # Spawn food
  if not food_spawn:
    food_pos = [random.randrange(1, 40)*10, random.randrange(1, 40)*10]
  food_spawn = True

  # Draw snake
  screen.fill((0,0,0))
  for pos in snake_body:
    pygame.draw.rect(screen, (200,200,200), pygame.Rect(pos[0], pos[1], 10, 10))

  # Draw food
  pygame.draw.rect(screen, (255,160,60), pygame.Rect(food_pos[0], food_pos[1], 10, 10))

  # Game over
  if snake_pos[0] < 0 or snake_pos[0] > 400 or snake_pos[1] < 0 or snake_pos[1] > 400:
    running = False
  if snake_pos in snake_body[1:]:
    running = False

  # Show score
  font = pygame.font.Font('freesansbold.ttf', 20)
  score_font = font.render('Score: ' + str(score), True, (255,255,255))
  screen.blit(score_font, (10,10))

  # Update display
  pygame.display.update()

#pygame.quit()