import user_image from './user-image.png';
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
import profile_img from './profile-img.png';
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

export const assets = {
    user_image,
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
    right_arrow_bold_dark
};

export const workData = [
    {
        title: 'STRETCH Robot Project',
        description: 'Dristributed autonomous robot collaboration with Moxie and STRETCH robots.',
        bgImage: '/Stretch_proj.gif',
    },
    {
        title: 'F1 Tenth Project',
        description: 'Seamless navigation with adaptive sensor switching based on lighting conditions.',
        bgImage: '/f1tenth_proj.gif',
    },
    {
        title: 'Universal Robotics Arm Project',
        description: 'Realistic drawing replication from digital images using OpenCV and a UR3e robot arm.',
        bgImage: '/UR3_proj.gif',
    },
    {
        title: 'GEM E4 Electric Vehicle Project',
        description: 'Persistent pedestrian detection and tracking through camera-LiDAR sensor fusion\.',
        bgImage: '/perception_proj.gif',
    },
]

export const serviceData = [
    { icon: assets.web_icon, title: 'Web design', description: 'Web development is the process of building, programming...', link: '' },
    { icon: assets.mobile_icon, title: 'Mobile app', description: 'Mobile app development involves creating software for mobile devices...', link: '' },
    { icon: assets.ui_icon, title: 'UI/UX design', description: 'UI/UX design focuses on creating a seamless user experience...', link: '' },
    { icon: assets.graphics_icon, title: 'Graphics design', description: 'Creative design solutions to enhance visual communication...', link: '' },
]

export const infoList = [
    { icon: assets.code_icon, iconDark: assets.code_icon_dark, title: 'Languages', description: 'Python, C++, Bash, PowerShell, Azure CLI, LabView' },
    { icon: assets.edu_icon, iconDark: assets.edu_icon_dark, title: 'Technologies', description: 'Deep Learning, Linux, Reinforcement Learning, NVIDIA Jetson, GEM E2, GEM E4, F1 Tenth, YOLO, Detic, SigLIP, UR3e' },
    { icon: assets.project_icon, iconDark: assets.project_icon_dark, title: 'Frameworks', description: 'PyTorch, OpenCV, ROS, ROS2, Gazebo, Gazebo Ign, Open AI, GEM Stack, Unity, Anaconda, Docker, Azure' }
];

export const toolsData = [
    assets.vscode, assets.firebase, assets.mongodb, assets.figma, assets.git
];