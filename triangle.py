def show():
    # Import a library of functions called 'pygame'
    import pygame
    from math import pi
     
    # Initialize the game engine
    pygame.init()
     
    # Define the colors we will use in RGB format
    BLACK = (  0,   0,   0)
    WHITE = (255, 255, 255)
    BLUE =  (  0,   0, 255)
    GREEN = (  0, 255,   0)
    RED =   (255,   0,   0)
     
    # Set the height and width of the screen
    size = [1200, 700]
    screen = pygame.display.set_mode(size)
     
    pygame.display.set_caption("Example code for the draw module")
     
    #Loop until the user clicks the close button.
    done = False
    clock = pygame.time.Clock()

    count=0;
     
    while not done:
     
        # This limits the while loop to a max of 10 times per second.
        # Leave this out and we will use all CPU we can.
        clock.tick(100)
        screen.fill(WHITE)
        
        #print count
        
        # if(count%2==0):
        BLACK = (  0,   0,   0)
    # else:
        RED = (  255,   0,   0)  #red
        GREEN = (  0,   255,   0)   #green
        BLUE = (  0,   0,   255)  #blue
        YELLOW = (  255,   255,   0)  #gray

        #For 100    
        									#center	     #left    #right
        #pygame.draw.polygon(screen, BLACK, [[600,50], [ 550,100], [650, 100]])
        
        #pygame.draw.polygon(screen, BLACK, [[600,650], [ 550,600], [650, 600]])

        #pygame.draw.polygon(screen, BLACK, [[100,300], [ 50,350], [100, 400]])
        
        #pygame.draw.polygon(screen, BLACK, [[1100,300], [ 1100,400], [1150, 350]])


        #For 120    
        if(count %2 == 0):
        	RED = GREEN = BLUE = YELLOW = BLACK	
    									#center	     #left    #right
        pygame.draw.polygon(screen, RED, [[600,20], [ 540,140], [660, 140]])  #up
        
        pygame.draw.polygon(screen, GREEN, [[600,680], [ 540,560], [660, 560]]) #down 

        pygame.draw.polygon(screen, BLUE, [[140,290], [ 20,350], [140, 410]]) #left
        
        pygame.draw.polygon(screen, YELLOW, [[1060,290], [ 1060,410], [1180, 350]]) #right



        pygame.display.flip()
        
        count=count+1;

        for event in pygame.event.get(): 
            if event.type == pygame.QUIT: 
                done=True

    pygame.quit()