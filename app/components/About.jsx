import { assets, infoList, toolsData } from '@/assets/assets'
import Image from 'next/image'
import React from 'react'

const About = ({isDarkMode}) => {
  return (
    <div id='about' className='w-full px-[12%] py-10 scroll-mt-20'>
      <h4 className='text-center mb-2 text-lg font-Ovo'>Introduction</h4>
      <h2 className='text-center text-5xl font-Ovo'>About me</h2>
      <div className='flex w-full flex-col lg:flex-row items-center gap-20 my-20'>
        <div  className='w-64 sm:w-80 rounded-3x1 max-w-none'>
          <Image src={assets.vikram_user_img} alt='user' className='w-full rounded-3xl' />
        </div>
        <div className='flex-1'>
          <p className='max-w-2x1 font-Ovo'>
          I’m a Master of Engineering candidate in Autonomy and Robotics at the University of Illinois Urbana-Champaign, passionate about bridging cutting-edge AI with 
          physical robotic systems to solve real-world challenges. My work revolves around computer vision, sensor fusion, and distributed autonomous systems, with a 
          focus on creating technologies that enhance human lives. 
          </p>
          <p className='mb-10 max-w-2x1 font-Ovo'>
          I am currently on the lookout for Summer Internship roles starting May 2025. Please feel free to reach out if you think I would be a good fit for your team!
          </p>
          <ul className='grid grid-cols-1 sm:grid-cols-3 gap-6 max-w-2xl'>
            {infoList.map(({icon, iconDark,title,description},index)=>(
              <li className='border-[0.5px] border-gray-400 rounded-xl p-6 cursor-pointer hover:bg-lightHover hover:-translate-y-1 
              duration-500 hover:shadow-black dark:border-white dark:hover:shadow-white dark:hover:bg-darkHover/50' key={index}>
                <Image src={isDarkMode ? iconDark : icon} alt={title} className='w-7 mt-3' />
                <h3 className='my-4 font-semibold text-gray-700 dark:text-white'>{title}</h3>
                <p className='text-gray-600 text-sm dark:text-white/80'>{description}</p>
              </li>
            ))}
          </ul>


        </div>
      </div>

    </div>
  )
}

export default About
