import vikram_user_img from './Vikram_user.jpg'
import code_icon from './code-icon.png';
import code_icon_dark from './code-icon-dark.png';
import edu_icon from './edu-icon.png';
import edu_icon_dark from './edu-icon-dark.png';
import project_icon from './project-icon.png';
import project_icon_dark from './project-icon-dark.png';
import vscode from './vscode.png';
import firebase from './firebase.png';
import figma from './figma.png';
import git from './git.png';
import mongodb from './mongodb.png';
import right_arrow_white from './right-arrow-white.png';
import logo from './logo.png';
import vikram_logo from './vikram-logo-light.png';
import vikram_logo_dark from './vikram-logo-dark.png';
import logo_dark from './logo_dark.png';
import mail_icon from './mail_icon.png';
import mail_icon_dark from './mail_icon_dark.png';
import profile_img from './profile-img.jpg';
import download_icon from './download-icon.png';
import hand_icon from './hand-icon.png';
import header_bg_color from './header-bg-color.png';
import moon_icon from './moon_icon.png';
import sun_icon from './sun_icon.png';
import arrow_icon from './arrow-icon.png';
import arrow_icon_dark from './arrow-icon-dark.png';
import menu_black from './menu-black.png';
import menu_white from './menu-white.png';
import close_black from './close-black.png';
import close_white from './close-white.png';
import web_icon from './web-icon.png';
import mobile_icon from './mobile-icon.png';
import ui_icon from './ui-icon.png';
import graphics_icon from './graphics-icon.png';
import right_arrow from './right-arrow.png';
import send_icon from './send-icon.png';
import right_arrow_bold from './right-arrow-bold.png';
import right_arrow_bold_dark from './right-arrow-bold-dark.png';
import linkedin from './linkdin.png';
import linkedin_dark from './linkdin_dark.png';
import github from './github.png';
import github_dark from './github_dark.png';

export const assets = {
    vikram_user_img,
    code_icon,
    code_icon_dark,
    edu_icon,
    edu_icon_dark,
    project_icon,
    project_icon_dark,
    vscode,
    firebase,
    figma,
    git,
    mongodb,
    right_arrow_white,
    logo,
    vikram_logo,
    vikram_logo_dark,
    logo_dark,
    mail_icon,
    mail_icon_dark,
    profile_img,
    download_icon,
    hand_icon,
    header_bg_color,
    moon_icon,
    sun_icon,
    arrow_icon,
    arrow_icon_dark,
    menu_black,
    menu_white,
    close_black,
    close_white,
    web_icon,
    mobile_icon,
    ui_icon,
    graphics_icon,
    right_arrow,
    send_icon,
    right_arrow_bold,
    right_arrow_bold_dark,
    linkedin_dark,
    linkedin,
    github,
    github_dark
};

export const workData = [
    {
        title: 'STRETCH Robot Project',
        description: 'Integrated STRETCH AI in STRETCH 2 at UIUC College of Applied Health Sciences and customized the repository to support the pick-and-place demo on GPUs with less than 8GB VRAM.',
        bgImage: '/Stretch_proj.gif',
        link:'https://github.com/lifehome-illinois/Stretch_AI_Illinois'
    },
    {
        title: 'F1 Tenth Project',
        description: 'Introduced adaptive sensor switching between camera and LiDAR based on lighting conditions. Optimized for NVIDIA Jetson, achieving high-speed lane detection using pure computer vision.',
        bgImage: '/f1tenth_proj.gif',
        link: 'https://www.youtube.com/watch?v=_oN69fMF7Dk',
    },
    {
        title: 'Universal Robotics Arm Project',
        description: 'Accurate replication of digital images into realistic drawings using OpenCV and a UR3e robotic arm, with a focus on improving drawing efficiency by retaining only essential features.',
        bgImage: '/UR3_proj.gif',
        link: 'https://uofi.box.com/s/kke8qw48yby7gts5eg5nbxk7hn3xkin6',
    },
    {
        title: 'Hyper-Realistic Gazebo Simulator',
        description: 'Designed a hyper-realistic simulation environment of UIUC Highbay for GEM E2/E4 autonomous vehicle and dynamic agent spawning from text-based scene descriptions.',
        bgImage: '/gazebo_gem.gif',
        link: 'https://github.com/harishkumarbalaji/POLARIS_GEM_Simulator'
    },
    {
        title: 'Persistent Pedestrian Tracker',
        description: 'Real-time, persistent pedestrian detection and tracking on a GEM E4 autonomous vehicle using techniques such as sensor fusion, YOLO, voxel downsampling, and 2D-3D projection/back-projection.',
        bgImage: '/perception_proj.gif',
        link: 'https://uofi.box.com/s/f4bd219vgkfieyz49dr2q30jx718238d'
    },
    {
        title: '6-Dof Pose Estimation Model',
        description: 'Developed a deep learning model combining YOLOv11 and ResNet to predict 6-DoF poses for robotic arm part-picking in warehouses, using RGB-D feature fusion via a dual-stream U-Net architecture.',
        bgImage: '/6dof.gif',
        link: 'https://github.com/nvikramraj/6-DoF-Pose-Model',
    },
]

export const timeData = [
    {
        title: 'Premier Cooperative Inc',
        time:'Aug 2025 - Present',
        role:'Founding Robotics Engineer, Contract',
        description: ['Designed a low cost solution to digitally override a joystick control box of a grain probe using a custom PCB, ESP32 microcontroller and NVIDIA Jetson enabling automation of grain sample collection from a truck.',
                      'Developed and containerized a ROS 2 pipeline using Docker that detects incoming grain trucks using YOLO, builds a spatial map using Intel Depth Sense camera and controls the grain probe to take grain samples every 45 seconds.',
                      'Building a web based control interface using React, FastAPI and Uvicorn to configure the system, calibrate cameras, override operations, and start/stop the robotic pipeline.'
        ],
    },
    {
        title: 'Kohler',
        time:'Jun 2025 - Aug 2025',
        role:'Robotics Perception Engineer, Apprentice',
        description: ['Developed a real time product packaging assistant and validation system for faucets using a lightweight Mask RCNN instance segmentation model, achieving 30 ms inference latency on NVIDIA Jetson.',
                      'Generated a synthetic dataset to augment missing faucet color and finish variations, resulting in a model achieving above 90% accuracy and mAP on real world inspection.'
        ],
    },
    {
        title: 'UIUC College of Veterinary Medicine',
        time:'Jun 2025 - Aug 2025',
        role:'Artificial Intelligence Engineer, Intern',
        description: ['Worked as a research assistant for Dr. Ben Blair, building AI training pipelines for swine wellness research.',
                      'Evaluated state of the art pose detection models trained on homogeneous and heterogeneous swine datasets to measure their impact on keypoint detection accuracy in real world farm environments.',
                      'Developed a web based application using React and Uvicorn to detect swine wellness through fecal matter classification.'
        ],
        
    },
    {
        title: 'UIUC College of Applied Health Sciences',
        time:'Jan 2025 - May 2025',
        role:'Robotics Perception Engineer, Intern',
        description: ['Optimized multimodal deep learning ROS2 pipeline for STRETCH Robot AI, enabling compatibility with low end GPUs by reducing VRAM usage from 12GB to 7GB.',
                      'Improved the reliability and safety of elderly assistive object pick and place tasks, doubling the pickup success rate from 20% to 40%.'
        ],
        
    },
    {
        title: 'University Of Illiois Urbana-Champaign',
        time:'Aug 2024 - Present',
        role:'M.Eng, Autonomy and Robotics',
        description: ['My course work includes working with mobile robotics and autonomous vehicles, currently focusing on developing my skill set on optimizing deep learning perception models and real time systems coordination.',
            'Related Course Work: Autonomous Vehicle System Engineering, Deep Learning with Computer Vision, Computer Vision, Principles of Safe Autonomy'
        ],
        
    },
    {
        title: 'Accenture',
        time:'Aug 2021 - Jul 2024',
        role:'Automation Engineer, Fulltime',
        description: ['Designed an automated system to monitor and maintain Honeywell Lenel S2 security systems deployed in Microsoft Datacenters globally, ensuring badge readers, CCTV cameras, and security applications operated reliably, and triggering alerts when an edge device fails.',
                      'Evaluated performance of cognitive vision models by assessing accuracy in detecting and reporting faulty surveillance cameras across Microsoft data centers.',
                      'Developed Azure CI/CD pipelines to provision, configure, and manage Docker containers, Azure virtual machines, and BareMetal servers, reducing manual setup and maintenance effort by 60%.'
        ],
        
    },
    {
        title: 'Nokia',
        time:'Feb 2021 - May 2021',
        role:'Embedded System Engineer, Intern',
        description: ['Developed an IoT device to detect obstacles blocking access to fire extinguishers and alert security in real time for 5G Radio Chip Manufacturing Factory.',
                      'Built a Python Flask backend with SQL database to communicate with ESP32 microcontroller to continuously update sensor data and transmit real time status updates to the security system.'
        ],
        
    },
    {
        title: 'B.S. Abdur Rahman Crescent Institute of Science & Technology',
        time:'Jul 2017 - Jun 2021',
        role:'B.Tech, Electronics and Communication Engineering',
        description: ['My course work included working with embedded systems, real time system processing and image processing',
            'Related Course Work: Real Time Embedded Systems, Image Processing, '
        ],
        
    }
]

export const serviceData = [
    { icon: assets.web_icon, title: 'Web design', description: 'Web development is the process of building, programming...', link: '' },
    { icon: assets.mobile_icon, title: 'Mobile app', description: 'Mobile app development involves creating software for mobile devices...', link: '' },
    { icon: assets.ui_icon, title: 'UI/UX design', description: 'UI/UX design focuses on creating a seamless user experience...', link: '' },
    { icon: assets.graphics_icon, title: 'Graphics design', description: 'Creative design solutions to enhance visual communication...', link: '' },
]

export const infoList = [
    { icon: assets.code_icon, iconDark: assets.code_icon_dark, title: 'Languages', description: 'Python, C++, Bash, PowerShell, Azure CLI, LabView, Embedded C, SQL, Matlab' },
    { icon: assets.edu_icon, iconDark: assets.edu_icon_dark, title: 'Technologies', description: 'Deep Learning, ViT, YOLO, Detic, SigLIP, FANUC, NVIDIA Jetson, GEM E2, GEM E4, F1 Tenth, UR3e, Reinforcement Learning ' },
    { icon: assets.project_icon, iconDark: assets.project_icon_dark, title: 'Frameworks', description: 'PyTorch, OpenCV, ROS, ROS2, ROBO Flow, Gazebo, Gazebo Ign, Open AI, GEM Stack, Unity, Anaconda, Docker, Azure, Linux' }
];

export const toolsData = [
    assets.vscode, assets.firebase, assets.mongodb, assets.figma, assets.git
];
