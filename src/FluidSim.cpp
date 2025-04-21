#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <camera.h>
#include <imgui.h>
#include <model.h>
#include <shader.h>
#include <texture.h>
#include <window.h>

#include <Particle.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <string>

// index of the current shader subroutine (= 0 in the beginning)
GLuint current_subroutine = 0;
GLuint frag_selector;
// a vector for all the shader subroutines names used and swapped in the application
vector<std::string> shaders;

// handle keystrokes
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode);

// setup functions
void initQuad(GLuint& quadVAO, GLuint& quadVBO);
void initFluidTextures();
void initParticleBuffers(GLuint& particleVBO, GLuint& particleVAO, int numParticles);

// render functions
void computeFluidThickness(Shader& thicknessShader, float& pWeight, GLuint particleVAO, int numParticles);
void renderParticles(Shader& particleShader, float target_density, ParticleSim& simulator, GLuint VAO);
void renderBlur(Shader& blurShader, GLuint quadVAO);
void renderPlane(Shader& planeShader, Model& planeModel, GLuint tex, GLuint quadVAO);
void renderDepth(Shader& depthShader, GLuint particleVAO, int numParticles);
void renderFluid(Shader& fluidShader, GLuint quadVAO);

// the name of the subroutines are searched in the shaders, and placed in the shaders vector (to allow shaders swapping)
void SetupShader(int shader_program);

// we initialize an array of booleans for each keyboard key
bool keys[1024];

glm::mat4 planeModelMatrix = glm::mat4(1.0f);
glm::mat3 planeNormalMatrix = glm::mat3(1.0f);

// parameters for time calculation (for animations)
GLfloat deltaTime = 0.0f;
GLfloat lastFrame = 0.0f;

// boolean to swap between particle rendering and fluid rendering
GLboolean fluidRenderingEnabled = GL_FALSE;

// fluid rendering parameters
float blurFalloff = 5.0;
float blurScale = 0.1;
float particleWeight = 0.015;
float rIndex = 1 / 1.33;
glm::vec4 fluidColor(0.3, 0.5, 1.0, 1.0);

// fluid rendering buffers and textures
GLuint particleDepthFBO, tempBlurFBO, blurDepthFBO, backgroundFBO, thicknessFBO;
GLuint particleDepthTexture, tempBlurTexture, blurDepthTexture, backgroundTexture, colorDepthTexture, thicknessTexture, textureCube;

// transform matrices
glm::mat4 projection;
glm::mat4 view;

/////////////////// MAIN function ///////////////////////
int main() {
    GLFWwindow* window = setup_window();
    glfwSetKeyCallback(window, key_callback);

    Shader particleShader = Shader("shaders/particle.vert", "shaders/particle.frag");
    Shader fluidShader = Shader("shaders/fluid.vert", "shaders/fluid.frag");
    Shader depthShader = Shader("shaders/depthRendering.vert", "shaders/depthRendering.frag");
    Shader blurShader = Shader("shaders/quad.vert", "shaders/bilateralFilter.frag");
    Shader thicknessShader = Shader("shaders/thickness.vert", "shaders/thickness.frag");
    Shader illuminationShader = Shader("shaders/basicIllumination.vert", "shaders/basicIllumination.frag");

    SetupShader(particleShader.Program);

    GLuint planeTexture = LoadTexture("../textures/UV_Grid_Sm.png");
    Model planeModel("../models/plane.obj");

    textureCube = LoadTextureCube("../textures/cube/tenerife2/");

    projection = glm::perspective(45.0f, (float)WIDTH / (float)HEIGHT, 0.1f, 100.0f);
    view = glm::mat4(1.0f);

    /**
     * GUI INIT
     */
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init();

    /**
     * PARTICLE SIMULATION PARAMETERS
     */
    float kernel_radius = 0.45;
    int numParticles = 50000;

    float3 volume = make_float3(12, 5, 8);
    float gas_constant = 0.33;
    float viscosity_mult = 6;
    float pressure_mult = 400;
    float target_density = 120;
    float particle_mass = 1;
    float timeStep = 1. / 60.;

    SimParams simParams = {
        numParticles,
        volume,
        gas_constant,
        target_density,
        pressure_mult,
        viscosity_mult,
        particle_mass};

    /**
     * SCREEN SPACE FLUID BUFFERS + INIT
     */

    GLuint quadVAO, quadVBO;
    initQuad(quadVAO, quadVBO);
    initFluidTextures();

    GLuint particleVBO, particleVAO;
    initParticleBuffers(particleVBO, particleVAO, numParticles);

    ParticleSim simulator = ParticleSim(simParams, particleVBO, kernel_radius);

    // Main render loop
    while (!glfwWindowShouldClose(window)) {
        bool reInitSim = false;
        GLfloat currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        glfwPollEvents();

        apply_camera_movements(camera, keys, deltaTime);
        view = camera.GetViewMatrix();

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        /***************************************************
         *  GUI setup
         */
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Disable these sliders while simulation is running
        ImGui::BeginDisabled(simulator.running_.load());
        if (ImGui::SliderFloat("SPH Kernel Radius", &kernel_radius, 0.15f, 1.f) ||
            ImGui::SliderInt("Num. Particles", &numParticles, 5000, 200000))
            reInitSim = true;  // Changing these sliders requires the simulator to be re-constructed
        ImGui::EndDisabled();

        ImGui::SliderFloat("perfect gas constant", &gas_constant, 0.0001f, 1.f);
        ImGui::SliderFloat("viscosity", &viscosity_mult, 0.0001f, 150.f);
        ImGui::SliderFloat("pressure_mult", &pressure_mult, 10.f, 600.f);
        ImGui::SliderFloat("target pressure", &target_density, 50.f, 1000.f);
        ImGui::SliderFloat("particle mass", &particle_mass, 0.0f, 10.f);

        simParams = {numParticles, volume, gas_constant,
                     target_density, pressure_mult, viscosity_mult, particle_mass};
        simulator.updateParams(simParams);

        ImGui::Separator();
        ImGui::Text("Simulation Timestep Control:");
        if (ImGui::Button("Adaptive", ImVec2(100, 50))) timeStep = -1;
        ImGui::SameLine();
        if (ImGui::Button("1x", ImVec2(100, 50))) timeStep = 1.0 / 60.0;
        ImGui::SameLine();
        if (ImGui::Button("0.5x", ImVec2(100, 50))) timeStep = 1.0 / 120.0;
        ImGui::SameLine();
        if (ImGui::Button("0.25x", ImVec2(100, 50))) timeStep = 1.0 / 240.0;
        simulator.setSimulationTimestep(timeStep);

        ImVec2 buttonSize(150, 50);
        ImGui::Separator();
        // Start simulation button (green)
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.4f, 0.8f, 0.3f, 1.0f));
        if (ImGui::Button("Start simulation", buttonSize) && !simulator.running_.load())
            simulator.startSimulation();

        ImGui::PopStyleColor();
        ImGui::SameLine();

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.7f, 0.7f, 1.f));
        if (ImGui::Button("Pause", ImVec2(50, 50)))
            simulator.pauseSimulation();
        ImGui::PopStyleColor();
        ImGui::SameLine();
        // Stop simulation button (red)
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.9f, 0.5f, 0.5f, 1.0f));
        // if the simulation is stopped or, a critical parameter has been changed, re-construct the simulation
        if (ImGui::Button("Stop simulation", buttonSize) || reInitSim) {
            simulator.~ParticleSim();
            glBindBuffer(GL_ARRAY_BUFFER, particleVBO);
            glBufferData(GL_ARRAY_BUFFER, numParticles * sizeof(Particle), NULL, GL_DYNAMIC_DRAW);
            glBindBuffer(GL_ARRAY_BUFFER, 0);

            new (&simulator) ParticleSim(simParams, particleVBO, kernel_radius);
        }
        ImGui::PopStyleColor();

        /*************************************************
         * Render functions
         */

        simulator.CUDAToOpenGLParticles();  // copy simulation data to particle VBO

        if (fluidRenderingEnabled) {
            renderDepth(depthShader, particleVAO, numParticles);

            // // --------- BLUR PASS --------------------
            ImGui::SliderFloat("blurFalloff", &blurFalloff, 0.1f, 150.f);
            ImGui::SliderFloat("blurScale", &blurScale, 0.01f, 1.f);

            renderBlur(blurShader, quadVAO);

            // RENDER THE PLANE UNDERNEATH THE FLUID
            renderPlane(illuminationShader, planeModel, planeTexture, quadVAO);

            // ----------------- THICKNESS MAP -----------------
            computeFluidThickness(thicknessShader, particleWeight, particleVAO, numParticles);

            // ----------------- FLUID RENDERING -----------------
            renderFluid(fluidShader, quadVAO);

        } else {
            renderParticles(particleShader, target_density, simulator, particleVAO);
        }

        /**************************************************
         * Final steps
         */
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    // we close and delete the created context
    glfwTerminate();
    return 0;
}

//////////////////////////////////////////
// callback for keyboard events
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);

    if (key == GLFW_KEY_F && action == GLFW_PRESS)
        fluidRenderingEnabled = !fluidRenderingEnabled;

    if ((key >= GLFW_KEY_1 && key <= GLFW_KEY_9) && action == GLFW_PRESS) {
        GLuint new_subroutine = (key - '0' - 1);
        frag_selector = new_subroutine;

        if (new_subroutine < shaders.size()) {
            current_subroutine = new_subroutine;
        }
    }

    if (action == GLFW_PRESS)
        keys[key] = true;
    else if (action == GLFW_RELEASE)
        keys[key] = false;
}

void SetupShader(int program) {
    int maxSub, maxSubU, countActiveSU;
    GLchar name[256];
    int len, numCompS;

    // global parameters about the Subroutines parameters of the system
    glGetIntegerv(GL_MAX_SUBROUTINES, &maxSub);
    glGetIntegerv(GL_MAX_SUBROUTINE_UNIFORM_LOCATIONS, &maxSubU);
    std::cout << "Max Subroutines:" << maxSub << " - Max Subroutine Uniforms:" << maxSubU << std::endl;

    // get the number of Subroutine uniforms (only for the Fragment shader, due to the nature of the exercise)
    // it is possible to add similar calls also for the Vertex shader
    glGetProgramStageiv(program, GL_FRAGMENT_SHADER, GL_ACTIVE_SUBROUTINE_UNIFORMS, &countActiveSU);

    // print info for every Subroutine uniform
    for (int i = 0; i < countActiveSU; i++) {
        // get the name of the Subroutine uniform (in this example, we have only one)
        glGetActiveSubroutineUniformName(program, GL_FRAGMENT_SHADER, i, 256, &len, name);
        // print index and name of the Subroutine uniform
        std::cout << "Subroutine Uniform: " << i << " - name: " << name << std::endl;

        // get the number of subroutines
        glGetActiveSubroutineUniformiv(program, GL_FRAGMENT_SHADER, i, GL_NUM_COMPATIBLE_SUBROUTINES, &numCompS);

        // get the indices of the active subroutines info and write into the array s
        int* s = new int[numCompS];
        glGetActiveSubroutineUniformiv(program, GL_FRAGMENT_SHADER, i, GL_COMPATIBLE_SUBROUTINES, s);
        std::cout << "Compatible Subroutines:" << std::endl;

        // for each index, get the name of the subroutines, print info, and save the name in the shaders vector
        for (int j = 0; j < numCompS; ++j) {
            glGetActiveSubroutineName(program, GL_FRAGMENT_SHADER, s[j], 256, &len, name);
            std::cout << "\t" << s[j] << " - " << name << "\n";
            shaders.emplace_back(name);
        }
        std::cout << std::endl;

        delete[] s;
    }
}

void initParticleBuffers(GLuint& particleVBO, GLuint& particleVAO, int numParticles) {
    glGenVertexArrays(1, &particleVAO);
    glBindVertexArray(particleVAO);

    glGenBuffers(1, &particleVBO);
    glBindBuffer(GL_ARRAY_BUFFER, particleVBO);
    glBufferData(GL_ARRAY_BUFFER, numParticles * sizeof(Particle), NULL, GL_DYNAMIC_DRAW);

    // Attribute 0: Particle position (float3)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)0);
    glEnableVertexAttribArray(0);

    // Attribute 1: Particle speed (float3)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)offsetof(Particle, velocity));
    glEnableVertexAttribArray(1);

    // Attribute 2: Particle density (float)
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)offsetof(Particle, density));
    glEnableVertexAttribArray(2);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void renderParticles(Shader& particleShader, float target_density, ParticleSim& simulator, GLuint VAO) {
    particleShader.Use();

    GLuint index = glGetSubroutineIndex(particleShader.Program, GL_FRAGMENT_SHADER, shaders[current_subroutine].c_str());
    // we activate the subroutine using the index (this is where shaders swapping happens)
    glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, 1, &index);

    glm::vec3 lightPos = glm::normalize(glm::vec3(1.0f, -1.0f, 1.0f));  // Light from top-right
    glUniform3fv(glGetUniformLocation(particleShader.Program, "lightPos"), 1, glm::value_ptr(lightPos));

    GLfloat pSize = 75.0f;
    glUniformMatrix4fv(glGetUniformLocation(particleShader.Program, "projectionMatrix"), 1, GL_FALSE, glm::value_ptr(projection));
    glUniformMatrix4fv(glGetUniformLocation(particleShader.Program, "viewMatrix"), 1, GL_FALSE, glm::value_ptr(view));
    glUniform1f(glGetUniformLocation(particleShader.Program, "pointSize"), pSize);

    GLfloat maxDensity = 1.5 * target_density;
    glUniform1f(glGetUniformLocation(particleShader.Program, "maxDensity"), maxDensity);

    glBindVertexArray(VAO);
    // Enable program point size so that the shader can set the point size per vertex
    glEnable(GL_PROGRAM_POINT_SIZE);
    // Render particles as points
    glDrawArrays(GL_POINTS, 0, simulator.getNumParticles());

    glDisable(GL_PROGRAM_POINT_SIZE);
    glBindVertexArray(0);
}

void renderPlane(Shader& planeShader, Model& planeModel, GLuint tex, GLuint quadVAO) {
    planeShader.Use();

    glBindFramebuffer(GL_FRAMEBUFFER, backgroundFBO);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUniformMatrix4fv(glGetUniformLocation(planeShader.Program, "projectionMatrix"), 1, GL_FALSE, glm::value_ptr(projection));
    glUniformMatrix4fv(glGetUniformLocation(planeShader.Program, "viewMatrix"), 1, GL_FALSE, glm::value_ptr(view));
    // PLANE
    // we activate the texture of the plane
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, tex);
    glUniform1i(glGetUniformLocation(planeShader.Program, "tex"), 1);
    glUniform1f(glGetUniformLocation(planeShader.Program, "repeat"), 2.0);

    planeModelMatrix = glm::mat4(1.0f);
    planeNormalMatrix = glm::mat3(1.0f);
    planeModelMatrix = glm::translate(planeModelMatrix, glm::vec3(7.5f, 0.0f, 4.0f));
    planeModelMatrix = glm::scale(planeModelMatrix, glm::vec3(4.0f, 1.0f, 4.0f));
    planeNormalMatrix = glm::inverseTranspose(glm::mat3(view * planeModelMatrix));
    glUniformMatrix4fv(glGetUniformLocation(planeShader.Program, "modelMatrix"), 1, GL_FALSE, glm::value_ptr(planeModelMatrix));
    glUniformMatrix3fv(glGetUniformLocation(planeShader.Program, "normalMatrix"), 1, GL_FALSE, glm::value_ptr(planeNormalMatrix));
    // we render the plane
    planeModel.Draw();

    glBindVertexArray(quadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void computeFluidThickness(Shader& thicknessShader, float& pWeight, GLuint particleVAO, int numParticles) {
    // render at half the resolution for efficiency
    glViewport(0, 0, WIDTH / 2, HEIGHT / 2);

    glBindFramebuffer(GL_FRAMEBUFFER, thicknessFBO);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // Enable additive blending.
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    glBlendEquation(GL_FUNC_ADD);

    thicknessShader.Use();

    ImGui::SliderFloat("particle_weight", &pWeight, 0.0f, 0.5f);

    glUniformMatrix4fv(glGetUniformLocation(thicknessShader.Program, "viewMatrix"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(thicknessShader.Program, "projectionMatrix"), 1, GL_FALSE, glm::value_ptr(projection));
    glUniform1f(glGetUniformLocation(thicknessShader.Program, "particleweight"), pWeight);

    glBindVertexArray(particleVAO);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glDrawArrays(GL_POINTS, 0, numParticles);
    glDisable(GL_PROGRAM_POINT_SIZE);

    glBindVertexArray(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDisable(GL_BLEND);

    // Restore full resolution viewport.
    glViewport(0, 0, WIDTH, HEIGHT);
    glClearColor(1.0, 1.0, 1.0, 1.0);
}

void renderDepth(Shader& depthShader, GLuint particleVAO, int numParticles) {
    depthShader.Use();
    glBindFramebuffer(GL_FRAMEBUFFER, particleDepthFBO);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    // Set matrices
    glUniformMatrix4fv(glGetUniformLocation(depthShader.Program, "viewMatrix"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(depthShader.Program, "projectionMat"), 1, GL_FALSE, glm::value_ptr(projection));

    // Render particles
    glBindVertexArray(particleVAO);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glDrawArrays(GL_POINTS, 0, numParticles);
    glDisable(GL_PROGRAM_POINT_SIZE);

    glBindVertexArray(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void renderBlur(Shader& blurShader, GLuint quadVAO) {
    blurShader.Use();
    // --------- VERTICAL BLUR PASS --------------------
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, colorDepthTexture);

    glUniform1i(glGetUniformLocation(blurShader.Program, "inputTexture"), 0);
    glUniform1f(glGetUniformLocation(blurShader.Program, "blurFalloff"), blurFalloff);
    glUniform1f(glGetUniformLocation(blurShader.Program, "blurScale"), blurScale);
    glUniform2f(glGetUniformLocation(blurShader.Program, "texelSize"), 1.0f / WIDTH, 1.0f / HEIGHT);
    glUniform2f(glGetUniformLocation(blurShader.Program, "direction"), 0, 1);

    glBindFramebuffer(GL_FRAMEBUFFER, tempBlurFBO);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glBindVertexArray(quadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // --------- HORIZONTAL BLUR PASS --------------------
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tempBlurTexture);

    glUniform2f(glGetUniformLocation(blurShader.Program, "direction"), 1, 0);
    glUniform1i(glGetUniformLocation(blurShader.Program, "inputTexture"), 0);

    glBindFramebuffer(GL_FRAMEBUFFER, blurDepthFBO);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glBindVertexArray(quadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void renderFluid(Shader& fluidShader, GLuint quadVAO) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    fluidShader.Use();

    glm::mat4 invView = glm::inverse(view);
    glm::mat4 invProjection = glm::inverse(projection);
    glUniformMatrix4fv(glGetUniformLocation(fluidShader.Program, "inverseView"), 1, GL_FALSE, glm::value_ptr(invView));
    glUniformMatrix4fv(glGetUniformLocation(fluidShader.Program, "inverseProjection"), 1, GL_FALSE, glm::value_ptr(invProjection));
    glUniformMatrix4fv(glGetUniformLocation(fluidShader.Program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
    glUniformMatrix4fv(glGetUniformLocation(fluidShader.Program, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniform3fv(glGetUniformLocation(fluidShader.Program, "cameraPos"), 1, glm::value_ptr(camera.Position));
    // refraction index uniform
    ImGui::SliderFloat("index of refraction", &rIndex, 0.0f, 1.5f);
    glUniform1f(glGetUniformLocation(fluidShader.Program, "refractIndex"), rIndex);

    ImGui::ColorEdit4("Fluid Color", glm::value_ptr(fluidColor));
    glUniform3fv(glGetUniformLocation(fluidShader.Program, "fluidColor"), 1, glm::value_ptr(fluidColor));
    glUniform1ui(glGetUniformLocation(fluidShader.Program, "fragSelector"), frag_selector);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, colorDepthTexture);
    glUniform1i(glGetUniformLocation(fluidShader.Program, "depthMap"), 0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, blurDepthTexture);
    glUniform1i(glGetUniformLocation(fluidShader.Program, "blurDepthMap"), 1);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, thicknessTexture);
    glUniform1i(glGetUniformLocation(fluidShader.Program, "thicknessMap"), 2);

    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, backgroundTexture);
    glUniform1i(glGetUniformLocation(fluidShader.Program, "backgroundTexture"), 3);

    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_CUBE_MAP, textureCube);
    glUniform1i(glGetUniformLocation(fluidShader.Program, "skybox"), 4);

    glBindVertexArray(quadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
}

void initFluidTextures() {
    bindFBOTexture(particleDepthFBO, colorDepthTexture);

    // we need to attach a depth texture to the FBO in order to enable z testing for the particles
    glGenTextures(1, &particleDepthTexture);
    glBindTexture(GL_TEXTURE_2D, particleDepthTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, WIDTH, HEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);

    glBindFramebuffer(GL_FRAMEBUFFER, particleDepthFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, particleDepthTexture, 0);

    // -------------- background texture --------------
    bindFBOTexture(backgroundFBO, backgroundTexture, 1.0, true);

    // -------------- thickness texture -----------------
    bindFBOTexture(thicknessFBO, thicknessTexture, 0.5, false);

    // Other textures
    bindFBOTexture(tempBlurFBO, tempBlurTexture);
    bindFBOTexture(blurDepthFBO, blurDepthTexture);
}

void initQuad(GLuint& quadVAO, GLuint& quadVBO) {
    float quadVertices[] = {
        // positions   // texCoords
        -1.0f, 1.0f, 0.0f, 1.0f,
        -1.0f, -1.0f, 0.0f, 0.0f,
        1.0f, -1.0f, 1.0f, 0.0f,

        -1.0f, 1.0f, 0.0f, 1.0f,
        1.0f, -1.0f, 1.0f, 0.0f,
        1.0f, 1.0f, 1.0f, 1.0f};
    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(5);
    glVertexAttribPointer(5, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(6);
    glVertexAttribPointer(6, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glBindVertexArray(0);
}