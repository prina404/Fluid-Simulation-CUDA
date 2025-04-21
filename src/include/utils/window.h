#pragma once

#include <GLFW/glfw3.h>
#include <camera.h>

#include <iostream>

void mouse_callback(GLFWwindow* window, double xpos, double ypos);

// dimensions of application's window
GLuint WIDTH = 2560, HEIGHT = 1440;

// we need to store the previous mouse position to calculate the offset with the current frame
GLfloat lastX, lastY;

// when rendering the first frame, we do not have a "previous state" for the mouse, so we need to manage this situation
bool firstMouse = true;

// rotation angle on Y axis
GLfloat orientationY = 0.0f;
// rotation speed on Y axis
GLfloat spin_speed = 30.0f;
// boolean to start/stop animated rotation on Y angle
GLboolean spinning = GL_TRUE;

// we create a camera. We pass the initial position as a parameter to the constructor. The last boolean tells that we want a camera "anchored" to the ground
Camera camera(glm::vec3(10.0f, 4.5f, 6.0f), GL_TRUE);
GLFWwindow* setup_window() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    // we set if the window is resizable
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

    // we create the application's window
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Fluid Simulator", nullptr, nullptr);
    if (!window) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        exit(-1);
    }
    glfwMakeContextCurrent(window);

    // we put in relation the window and the callbacks
    glfwSetCursorPosCallback(window, mouse_callback);

    // we disable the mouse cursor
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

    // GLAD tries to load the context set by GLFW
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize OpenGL context" << std::endl;
        exit(-1);
    }

    // we define the viewport dimensions
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    glViewport(0, 0, width, height);

    // we enable Z test
    glEnable(GL_DEPTH_TEST);
    // white background
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    return window;
}

//////////////////////////////////////////
// callback for mouse events
void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    // Only process camera movement if the left mouse button is pressed.
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) != GLFW_PRESS) {
        firstMouse = true;
        return;
    }

    // Initialize previous mouse coordinates if this is the first frame
    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    // Calculate offset from previous mouse position
    GLfloat xoffset = xpos - lastX;
    GLfloat yoffset = lastY - ypos;

    // Update last positions for the next frame
    lastX = xpos;
    lastY = ypos;

    // Update the camera based on mouse movement
    camera.ProcessMouseMovement(xoffset, yoffset);
}