  import { assets } from '@/assets/assets'
import Image from 'next/image'
import React from 'react'
import { motion } from "motion/react"

const Header = ({isDarkMode}) => {
  return (

    <div className='w-11/12 max-w-3xl text-center mx-auto min-h-screen flex flex-col items-center justify-start gap-4 pt-24 pb-12'>
      <motion.div 
        initial={{scale: 0}}
        whileInView={{scale: 1}}
        transition={{duration:0.8, type:'spring', stiffness: 100}}
        className="mt-8"
      >
        <Image 
          src={assets.profile_img} 
          alt='Profile' 
          className='rounded-full w-32 h-32 object-cover'
        />
      </motion.div>
      <h3 className='flex items-end gap-2 text-xl md:text-2xl mb-3 font-Ovo'>Vikram Raj Nagoor Kani </h3>
      <h1 className='text-3xl sm:text-6xl lg:text-[66px] font-Ovo'>M.Eng Autonomy and Robotics, University of Illinois Urbana-Champaign </h1>
      <p className='max-w-2xl mx-auto font-Ovo'>Robotics Engineer</p>
      <div className='flex flex-col sm:flex-row items-center gap-4 mt-4'>
        <a href="#contact" className='px-10 py-3 border border-white rounded-full bg-black text-white flex items-center gap-2 dark:bg-transparent'>
          Contact Me <Image src={assets.right_arrow_white} alt='' className='w-4' /></a>
        <a href="/Vikram_Resume.pdf" download className='px-10 py-3 border rounded-full border-gray-500 flex items-center gap-2 bg-white dark:text-black' >
        My Resume <Image src={assets.download_icon} alt='' className='w-4' /></a>
      </div>
      
      
      <div className='flex flex-col sm:flex-row items-center gap-4 mt-4'>

        <a href="https://www.linkedin.com/in/nvikramraj/" className='p-3 
          transition-all duration-300 hover:scale-105'>
          <Image src={isDarkMode ? assets.linkedin_dark : assets.linkedin} alt='LinkedIn' className='w-12 h-12 sm:w-12 sm:h-12 object-contain dark:invert'/>
        </a>
        <a href="https://github.com/nvikramraj" className='
            p-1 rounded-full bg-white dark:bg-white border-2 border-gray-300 dark:border-gray-600 hover:bg-gray-200 dark:hover:bg-gray-200 
            transition-all duration-300 hover:scale-105 w-15 h-15 flex items-center justify-center'>
            <Image src={isDarkMode ? assets.github : assets.github_dark} alt='GitHub' className='w-12 h-12 object-contain'/>
        </a>

      </div>


      
    </div>
  )
}

export default Header
